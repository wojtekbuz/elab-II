import csv
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

with open("case I/supermarket.csv", "r", newline="") as csvfile:
    reader = csv.reader(csvfile)
    rows = []

    for row in reader:
        row = row[:]
        rows.append(row)

df = pd.DataFrame(rows[:1000])

department_spend = {}
department_items = {}
department_time = {}
total_items = 0
total_shop_time = 0

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    previous_department = None
    previous_time = 0
    total_spend = 0
    items_bought = 0

    # Iterate over each tuple in the row
    for item in row:
        # Split the tuple
        if item and len(item.split()) == 3:
            department, time, price = item.split()
            time = int(time)
            price = float(price)

        else:
            pass

        if department in department_spend:
            department_spend[department].append(price)
            department_items[department] += 1
        else:
            department_spend[department] = [price]
            department_items[department] = 1

        if department in department_time:
            department_time[department].append(time - previous_time)
        else:
            department_time[department] = [time - previous_time]

        total_spend += price
        items_bought += 1

        previous_department = department
        previous_time = time

    total_items += items_bought
    total_shop_time += previous_time

total_items_bought = sum(department_items.values())

average_spend_per_department = {
    dept: sum(spend) / department_items[dept]
    for dept, spend in department_spend.items()
}
average_spend_per_item_per_department = {
    dept: sum(spend) / total_items_bought for dept, spend in department_spend.items()
}
average_travel_time = {
    dept: sum(time) / department_items[dept] for dept, time in department_time.items()
}
average_items_per_department = {
    dept: department_items[dept] / len(department_items) for dept in department_items
}

new_df = pd.DataFrame(
    {
        "Department": list(department_spend.keys()),
        "Average Spend per Department": list(average_spend_per_department.values()),
        "Average Spend per Item per Department": list(
            average_spend_per_item_per_department.values()
        ),
        "Average Travel Time between Departments": list(average_travel_time.values()),
        "Average Items Bought per Department": list(
            average_items_per_department.values()
        ),
    }
)

# ASSOCIATION RULES
transactions = []
for index, row in df.iterrows():
    transaction = []
    for item in row:
        if item and len(item.split()) == 3:
            department, time, price = item.split()
            time = int(time)
            price = float(price)
            transaction.append(department)
        else:
            transaction.append(0)
    transactions.append(transaction)

transaction_df = pd.DataFrame(transactions)

encoder = TransactionEncoder()

transaction_df = transaction_df.applymap(str)

association_data = encoder.fit_transform(transaction_df.values.tolist())

association_df = pd.DataFrame(association_data, columns=encoder.columns_)

association_df = association_df.astype(int)

association_df = apriori(association_df, min_support = 0.01, use_colnames = True, verbose = 1)
association_df_filtered = association_df[association_df['support'] < 0.1]

print(association_df_filtered)

# df_ar = association_rules(association_df_filtered, metric = "confidence", min_threshold = 0.6)