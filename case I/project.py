import csv
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from prefixspan import PrefixSpan
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


class DataMiner:
    def __init__(self, data_file):
        self.data_file = data_file

    def load_data(self):
        with open(self.data_file, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
        df = pd.DataFrame(rows)

        transactions = {}
        departments = {}
        for index, row in df.iterrows():
            for item in row:
                if item and len(item.split()) == 3:
                    department, time, price = item.split()
                    if index not in transactions:
                        transactions[index] = {"departments": [], "times": [], "prices": []}
                    if department not in departments:
                        departments[department] = {"times": [], "prices": []}

                    transactions[index]["departments"].append(department)
                    transactions[index]["times"].append(float(time))
                    transactions[index]["prices"].append(float(price))

                    departments[department]["times"].append(float(time))
                    departments[department]["prices"].append(float(price))

        return transactions, departments

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

    def find_time_outliers(self, lower_percentile, upper_percentile, k, threshold_percentile):
        transactions, departments = self.load_data()
        time_bounds = {}
        total_time_spent = {}
        total_items = {}

        for department, data in departments.items():
            time_values = data["times"]
            time = np.array(time_values)

            lower_bound = float(np.percentile(time, lower_percentile))
            upper_bound = float(np.percentile(time, upper_percentile))

            time_bounds[department] = (lower_bound, upper_bound)
        
        for transaction, data in transactions.items():
            departments = data["departments"]
            price_values = data["times"]
            total_time_spent[transaction] = sum(price_values)
            total_items[transaction] = len(departments)

        X = np.array([[total_time_spent[transaction], total_items[transaction]] for transaction in transactions])

        kmeans = KMeans(n_clusters = k, random_state=1)
        kmeans.fit(X)

        cluster_labels = kmeans.labels_
        cluster_centroids = kmeans.cluster_centers_

        clusters = {i: [] for i in range(len(cluster_centroids))}

        distances = pairwise_distances_argmin_min(X, cluster_centroids)[1]

        threshold = np.percentile(distances, threshold_percentile)

        # Assign transactions to clusters
        for transaction, _ in total_time_spent.items():
            cluster_label = cluster_labels[transaction]
            clusters[cluster_label].append(transaction)

        plt.figure(figsize=(10, 6))
        for label, transactions in clusters.items():
            plt.scatter([total_time_spent[t] for t in transactions], [total_items[t] for t in transactions], label=f'Cluster {label}', alpha=0.5)
        plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], s=300, c='red', marker='X', edgecolors='black', label='Centroids')
        plt.title('Clusters and Centroids')
        plt.xlabel('Total Time Spent')
        plt.ylabel('Total Items')
        plt.legend()
        plt.grid(True)

        print("Time bounds and centroids determined.")

        return time_bounds, cluster_centroids, threshold

    def find_price_outliers(self, k, threshold_percentile):
        transactions, departments = self.load_data()
        total_price_spent = {}
        total_items = {}

        # TRANSACTION-LEVEL CLUSTERING BASED ON TOTAL SPENT AND TOTAL ITEMS
        for transaction, data in transactions.items():
            departments = data["departments"]
            price_values = data["prices"]
            total_price_spent[transaction] = sum(price_values)
            total_items[transaction] = len(departments)

        X = np.array([[total_price_spent[transaction], total_items[transaction]] for transaction in transactions])

        kmeans = KMeans(n_clusters = k, random_state=1)
        kmeans.fit(X)

        cluster_labels = kmeans.labels_
        cluster_centroids = kmeans.cluster_centers_

        clusters = {i: [] for i in range(len(cluster_centroids))}

        distances = pairwise_distances_argmin_min(X, cluster_centroids)[1]

        threshold = np.percentile(distances, threshold_percentile)

        # Assign transactions to clusters
        for transaction, _ in total_price_spent.items():
            cluster_label = cluster_labels[transaction]
            clusters[cluster_label].append(transaction)

        plt.figure(figsize=(10, 6))
        for label, transactions in clusters.items():
            plt.scatter([total_price_spent[t] for t in transactions], [total_items[t] for t in transactions], label=f'Cluster {label}', alpha=0.5)
        plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], s=300, c='red', marker='X', edgecolors='black', label='Centroids')
        plt.title('Clusters and Centroids')
        plt.xlabel('Total Price Spent')
        plt.ylabel('Total Items')
        plt.legend()
        plt.grid(True)

        print("Price centroids determined.")
    
        return cluster_centroids, threshold