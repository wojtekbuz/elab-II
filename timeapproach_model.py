import csv
import random

# Load the upper decision values for each department
upper_decision_values = {
    1: 235.0,
    2: 357.0,
    3: 128.0,
    4: 198.0,
    5: 63.0,
    6: 228.0,
    7: 93.0,
    8: 211.0,
    9: 321.7,
    10: 199.0,
    11: 75.0,
    12: 221.0,
    13: 199.0,
    14: 142.0,
    15: 165.0,
    16: 216.0,
    17: 168.0,
    18: 267.0
}

# Load the lower decision values for each department
lower_decision_values = {
    1: 29.0,
    2: 50.0,
    3: 29.0,
    4: 39.1,
    5: 9.0,
    6: 24.0,
    7: 9.0,
    8: 32.0,
    9: 23.0,
    10: 31.0,
    11: 17.0,
    12: 35.0,
    13: 38.0,
    14: 25.0,
    15: 22.0,
    16: 34.0,
    17: 23.0,
    18: 42.0
}

# Load the test transactions
test_transactions = []
transaction_ids = []

with open("case12.csv", "r", newline="") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        transaction = []
        transaction_info = row[0].split(":")
        transaction_index = transaction_info[0]
        first_transaction = transaction_info[1]
        if first_transaction:
            department, time_spent, _ = first_transaction.split()
            transaction.append((department, float(time_spent)))
        for item in row:
            if item and len(item.split()) == 3:
                department, time_spent, _ = item.split()
                transaction.append((department, float(time_spent)))
        if transaction:
            test_transactions.append(transaction)
            transaction_ids.append(transaction_index)


# Function to check if a transaction is potentially fraudulent based on total time spent
def is_potentially_fraudulent(transaction):
    below_lower_count = 0
    above_upper_count = 0

    for department, time_spent in transaction:
        # Check if time spent exceeds upper decision value
        if time_spent > upper_decision_values.get(int(department), float('inf')):
            above_upper_count += 1

        # Check if time spent falls below lower decision value
        if time_spent < lower_decision_values.get(int(department), float('-inf')):
            below_lower_count += 1

    # Flag transaction if at least 5 departments exceed upper decision value or fall below lower decision value
    if above_upper_count >= 4 and below_lower_count >= 1:
        return True
    return False


# Identify potentially fraudulent transactions
flagged_transactions = []

for idx, transaction in zip(transaction_ids, test_transactions):
    if is_potentially_fraudulent(transaction):
        flagged_transactions.append(idx)

# Shuffle the list of flagged transactions
random.shuffle(flagged_transactions)

# Output flagged transactions
print("Flagged Transactions:")
for idx in flagged_transactions[:150]:
    print(idx.split()[1])  # Print only the number part of the transaction ID

print("Total flagged transactions:", len(flagged_transactions))



