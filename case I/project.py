import csv
import pandas as pd
import numpy as np
import pickle
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from prefixspan import PrefixSpan


class DataMiner:
    def __init__(self, data_file):
        self.data_file = data_file

    def load_data(self):
        with open(self.data_file, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
        df = pd.DataFrame(rows)

        transactions = {}
        for index, row in df.iterrows():
            for item in row:
                if item and len(item.split()) == 3:
                    department, time, price = item.split()
                    if department not in transactions:
                        transactions[department] = {"time": [], "price": []}
                    transactions[department]["time"].append(float(time))
                    transactions[department]["price"].append(float(price))
        return transactions

    def mine_association_rules(self, support, lift):
        try:
            with open("case I/association_rules.pkl", "rb") as f:
                df_ar = pickle.load(f)
                print("Association rules read from file.")
        except FileNotFoundError:
            transactions = self.load_data()
            encoder = TransactionEncoder()
            association_data = encoder.fit_transform(transactions)
            association_df = pd.DataFrame(association_data, columns=encoder.columns_)
            association_df = apriori(
                association_df, min_support=0.001, use_colnames=True, low_memory=True
            )
            df_ar = association_rules(
                association_df, metric="confidence", min_threshold=0.75
            )
            with open("case I/association_rules.pkl", "wb") as f:
                pickle.dump(df_ar, f)

        association_df_filtered = df_ar[
            (df_ar["support"] >= support)
            & (df_ar["lift"] >= lift)
            & (df_ar["antecedents"].apply(lambda x: len(x)) > 1)
        ]

        sorted_rules = association_df_filtered.sort_values(
            by=["support", "confidence", "lift"], ascending=[False, False, False]
        )

        return sorted_rules

    def mine_sequential_rules(self, threshold):
        try:
            with open("case I/sequential_rules.pkl", "rb") as f:
                frequent_sequences = pickle.load(f)
                print("Sequential rules read from file.")
        except FileNotFoundError:
            transactions = self.load_data()
            frequent_sequences = PrefixSpan(transactions)
            frequent_sequences = list(frequent_sequences.frequent(1, closed=True))
            with open("case I/sequential_rules.pkl", "wb") as f:
                pickle.dump(frequent_sequences, f)

        filtered_sequences = [
            seq
            for support, seq in frequent_sequences
            if support >= threshold and len(seq) > 1
        ]

        return filtered_sequences

    def find_time_outliers(self, lower_percentile, upper_percentile):
        transactions = self.load_data()
        time_bounds = {}

        for department, data in transactions.items():
            time_values = data["time"]
            time = np.array(time_values)

            lower_bound = float(np.percentile(time, lower_percentile))
            upper_bound = float(np.percentile(time, upper_percentile))

            time_bounds[department] = (lower_bound, upper_bound)

        print("Time bounds determined.")
        return time_bounds

    def find_price_outliers(self, lower_percentile, upper_percentile):
        transactions = self.load_data()
        price_bounds = {}

        for department, data in transactions.items():
            price_values = data["price"]
            prices = np.array(price_values)

            lower_bound = float(np.percentile(prices, lower_percentile))
            upper_bound = float(np.percentile(prices, upper_percentile))

            price_bounds[department] = (lower_bound, upper_bound)

        print("Price bounds determined.")
        return price_bounds
