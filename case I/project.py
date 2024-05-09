import csv
import pandas as pd
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
        return pd.DataFrame(rows)

    def mine_association_rules(self, df, support, lift):
        try:
            with open("case I/association_rules.pkl", "rb") as f:
                df_ar = pickle.load(f)
                print("Association rules read from file.")
        except FileNotFoundError:
            transactions = self._process_transactions(df)
            encoder = TransactionEncoder()
            association_data = encoder.fit_transform(transactions)
            association_df = pd.DataFrame(association_data, columns=encoder.columns_)
            association_df = apriori(
                association_df, min_support=0.001, use_colnames=True, low_memory=True
            )
            df_ar = association_rules(association_df, metric="confidence", min_threshold=0.75)
            with open("case I/association_rules.pkl", "wb") as f:
                pickle.dump(df_ar, f)

        association_df_filtered = df_ar[
        (df_ar["support"] >= support) & (df_ar["lift"] >= lift) &
        (df_ar["antecedents"].apply(lambda x: len(x)) > 1)
    ]
        
        sorted_rules = association_df_filtered.sort_values(
        by=["support", "confidence", "lift"], ascending=[False, False, False]
    )
        
        return sorted_rules

    def mine_sequential_rules(self, df, threshold):
        try:
            with open("case I/sequential_rules.pkl", "rb") as f:
                frequent_sequences = pickle.load(f)
                print("Sequential rules read from file.")
        except FileNotFoundError:
            transactions = self._process_transactions(df)
            frequent_sequences = PrefixSpan(transactions)
            frequent_sequences = list(frequent_sequences.frequent(1, closed=True))
            with open("case I/sequential_rules.pkl", "wb") as f:
                pickle.dump(frequent_sequences, f)
        
        filtered_sequences = [seq for support, seq in frequent_sequences if support >= threshold and len(seq) > 1]

        return filtered_sequences

    def _process_transactions(self, df):
        transactions = []
        for index, row in df.iterrows():
            transaction = []
            for item in row:
                if item and len(item.split()) == 3:
                    department, time, price = item.split()
                    transaction.append(str(department))
            transactions.append(transaction)
        return transactions

if __name__ == "__main__":
    data_miner = DataMiner("case I/supermarket.csv")
    df = data_miner.load_data()