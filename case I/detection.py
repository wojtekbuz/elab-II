from project import *

with open("case I/case0.csv", "r", newline="") as csvfile:
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

flagged_transactions = []
for idx, transaction in zip(transaction_ids, test_transactions):
    # Assume the transaction is fraudulent
    fraudulent = True
    for _, rule in sorted_rules.iterrows():
        antecedent = set(rule["antecedents"])
        consequent = set(rule["consequents"])
        # Check if the antecedent and consequent are in the transaction
        if antecedent.issubset(set(transaction)) and consequent.issubset(
            set(transaction)
        ):
            # If the antecedent and consequent are in the transaction, unflag the transaction
            fraudulent = False

    if fraudulent == True:
        flagged_transactions.append((idx, transaction))

print("Flagged Transactions:")
for idx, transaction in flagged_transactions:
    print(f"{idx}")

print(len(flagged_transactions))
