import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def identify_counts_outliers(counts_per_customer, department=None):
    # Filter out customers who bought 0 items in the department
    counts_per_customer_filtered = counts_per_customer[counts_per_customer > 0]

    # Calculate decision values
    decision_value_low = np.percentile(counts_per_customer_filtered, 10)
    decision_value_high = np.percentile(counts_per_customer_filtered, 90)

    # Identify outliers based on counts thresholds
    outliers_low_counts = np.where(counts_per_customer_filtered < decision_value_low)[0]
    outliers_high_counts = np.where(counts_per_customer_filtered > decision_value_high)[0]

    print("Number of low counts outliers:", len(outliers_low_counts))
    print("Number of high counts outliers:", len(outliers_high_counts))
    print("Decision value for 10th percentile:", decision_value_low)
    print("Decision value for 90th percentile:", decision_value_high)

    # Plot histogram and box plot
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(counts_per_customer_filtered, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Items Bought')
    plt.ylabel('Frequency')
    plt.title(f'Item Count Distribution for {department}')

    plt.subplot(1, 2, 2)
    plt.boxplot(counts_per_customer_filtered, vert=False)
    plt.xlabel('Number of Items Bought')
    plt.title(f'Boxplot for {department}')

    plt.tight_layout()
    plt.show()


# Read the CSV file, skipping the first row
df = pd.read_csv("supermarket_fixed.csv", sep=';', skiprows=1, header=None)

# Convert all values to integers, keeping decimals
df = df.apply(lambda x: pd.to_numeric(x))

# Extract transaction data (products) for each customer
transaction_data = df.iloc[:, 1:]

def calculate_department_counts(transaction_data):
    department_counts_list = []

    # Iterate over each row in the transaction data
    for idx, row in transaction_data.iterrows():
        customer_counts = [0] * 18  # Initialize counts for each department to 0
        dept_idx = 0  # Initialize department index
        # Iterate over each value in the row
        for val in row:
            if pd.notnull(val):
                if dept_idx % 3 == 0:  # Department number
                    dept = int(val)
                elif dept_idx % 3 == 2:  # Product count
                    if 1 <= dept <= 18:
                        customer_counts[dept - 1] += 1
                dept_idx += 1
        department_counts_list.append(customer_counts)

    return pd.DataFrame(department_counts_list, columns=range(1, 19))


# Calculate department counts for each customer
department_counts_df = calculate_department_counts(transaction_data)

# Identify outliers for counts in each department
for department in range(1, 19):
    counts_data = department_counts_df[department]
    print(f"Department: {department}")
    identify_counts_outliers(counts_data, f"Department {department}")

