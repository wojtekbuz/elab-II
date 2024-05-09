import csv
import random
from project import DataMiner


class FraudDetector:
    def __init__(self, data_miner):
        self.data_miner = data_miner

    def load_transactions(self, filepath):
        with open(filepath, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            test_transactions = []
            transaction_ids = []
            for row in reader:
                transaction = []
                transaction_info = row[0].split(":")
                transaction_index = transaction_info[0]
                first_transaction = transaction_info[1]
                if first_transaction:
                    department, _, _ = first_transaction.split()
                    transaction.append(department)
                for item in row:
                    if item and len(item.split()) == 3:
                        department, _, _ = item.split()
                        transaction.append(department)
                if transaction:
                    test_transactions.append(transaction)
                    transaction_ids.append(transaction_index)

            return test_transactions, transaction_ids

    def detect_fraud(self, test_transactions, transaction_ids, score_threshold):
        flagged_transactions = []

        sorted_rules = data_miner.mine_association_rules(df, support=0.33, lift=1.3)
        print(sorted_rules)
        frequent_sequences = data_miner.mine_sequential_rules(df, threshold=460)
        print(frequent_sequences)

        for idx, transaction in zip(transaction_ids, test_transactions):
            fraud_score = 2
            for _, rule in sorted_rules.iterrows():
                antecedent = set(rule["antecedents"])
                consequent = set(rule["consequents"])
                if antecedent.issubset(set(transaction)) and not consequent.issubset(
                    set(transaction)
                ):
                    fraud_score -= 1
                    break

            for seq in frequent_sequences:
                for item in range(len(transaction) - len(seq) + 1):
                    if transaction[item : item + len(seq)] == seq:
                        fraud_score -= 1
                        break

            if fraud_score == score_threshold:
                flagged_transactions.append((idx, transaction))

        return flagged_transactions

    def print_flagged_transactions(self, flagged_transactions):
        if len(flagged_transactions) == 0:
            print("No flagged transactions")
        elif len(flagged_transactions) <= 150:
            print(f"{len(flagged_transactions)} Flagged Transactions")
            for idx, _ in flagged_transactions:
                transaction_number = idx.split()[1]
                print(f"{transaction_number}")
        else:
            random_selected_transactions = random.sample(flagged_transactions, 150)
            print(
                f"{len(random_selected_transactions)} / {len(flagged_transactions)} Flagged Transactions"
            )
            for idx, _ in random_selected_transactions:
                transaction_number = idx.split()[1]
                print(f"{transaction_number}")


if __name__ == "__main__":
    data_miner = DataMiner("case I/supermarket.csv")
    df = data_miner.load_data()

    fraud_detector = FraudDetector(data_miner)
    test_transactions, transaction_ids = fraud_detector.load_transactions(
        "case I/case9.csv"
    )
    flagged_transactions = fraud_detector.detect_fraud(
        test_transactions, transaction_ids, score_threshold=2
    )
    fraud_detector.print_flagged_transactions(flagged_transactions)
