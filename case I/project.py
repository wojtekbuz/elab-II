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

try:
    with open("case I/association_rules.pkl", "rb") as f:
        df_ar = pickle.load(f)
        print("ar read done")
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
        print("sr read done")
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
    frequent_sequences = list(frequent_sequences.frequent(1, closed=True))

    with open("case I/sequential_rules.pkl", "wb") as f:
        pickle.dump(frequent_sequences, f)

association_df_filtered = df_ar[
    (df_ar["support"] > 0.1) & (df_ar["zhangs_metric"] > 0.6)
]
sorted_rules = association_df_filtered.sort_values(
    by=["support", "confidence", "zhangs_metric"], ascending=[False, False, False]
)

print(sorted_rules.head(20))
print(len(sorted_rules))

for support, seq in frequent_sequences:
    if support < 20 and len(seq) <= 2 and len(seq) > 1:
        print(support, seq)

for support, seq in frequent_sequences:
    if support > 400 and len(seq) <= 3 and len(seq) > 1:
        print(support, seq)

# In fraud detection or anomaly detection scenarios, such association rules with low support but high confidence and
# significance (as indicated by Zhang's metric) can be valuable because they represent unusual or suspicious patterns that deviate from the norm.
# These rules may highlight potentially fraudulent behavior or rare but meaningful patterns in the data.
