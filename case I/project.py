import csv
import pandas as pd
import pickle
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from prefixspan import PrefixSpan

with open("case I/supermarket.csv", "r", newline="") as csvfile:
    reader = csv.reader(csvfile)
    rows = []

    for row in reader:
        row = row[:]
        rows.append(row)

df = pd.DataFrame(rows)

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

try:
    with open("case I/association_rules.pkl", "rb") as f:
        df_ar = pickle.load(f)
except FileNotFoundError:
    # ASSOCIATION RULES
    transactions = []
    for index, row in df.iterrows():
        transaction = []
        for item in row:
            if item and len(item.split()) == 3:
                department, time, price = item.split()
                time = int(time)
                price = float(price)
                transaction.append(str(department))
        transactions.append(transaction)

    encoder = TransactionEncoder()
    association_data = encoder.fit_transform(transactions)

    association_df = pd.DataFrame(association_data, columns=encoder.columns_)

    association_df = apriori(
        association_df, min_support=0.001, use_colnames=True, low_memory=True
    )
    df_ar = association_rules(association_df, metric="confidence", min_threshold=0.75)

    with open("case I/association_rules.pkl", "wb") as f:
        pickle.dump(df_ar, f)


try:
    with open("case I/sequential_rules.pkl", "rb") as f:
        frequent_sequences = pickle.load(f)
except FileNotFoundError:
    # SEQUENTIAL RULES
    transactions = []
    for index, row in df.iterrows():
        transaction = []
        for item in row:
            if item and len(item.split()) == 3:
                department, time, price = item.split()
                time = int(time)
                price = float(price)
                transaction.append(str(department))
        transactions.append(transaction)

    frequent_sequences = PrefixSpan(transactions)

    with open("case I/sequential_rules.pkl", "wb") as f:
        pickle.dump(frequent_sequences, f)


association_df_filtered = df_ar[
    (df_ar["support"] < 0.05) & (df_ar["zhangs_metric"] > 0.8)
]

print(association_df_filtered)
print(len(association_df_filtered))

for support, seq in frequent_sequences.frequent(1, closed=True):
    if support < 5:
        print(support, seq)

# In fraud detection or anomaly detection scenarios, such association rules with low support but high confidence and
# significance (as indicated by Zhang's metric) can be valuable because they represent unusual or suspicious patterns that deviate from the norm.
# These rules may highlight potentially fraudulent behavior or rare but meaningful patterns in the data.
