import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def identify_spending_outliers(spending_per_customer, department=None):
    # Filter out customers who spent 0 in the department
    spending_per_customer_filtered = spending_per_customer[spending_per_customer > 0]

    # Calculate decision values
    decision_value_low = np.percentile(spending_per_customer_filtered, 10)
    decision_value_high = np.percentile(spending_per_customer_filtered, 90)

    # Identify outliers based on spending thresholds
    outliers_low_spending = np.where(spending_per_customer_filtered < decision_value_low)[0]
    outliers_high_spending = np.where(spending_per_customer_filtered > decision_value_high)[0]

    print("Number of low spending outliers:", len(outliers_low_spending))
    print("Number of high spending outliers:", len(outliers_high_spending))
    print("Decision value for 10th percentile:", decision_value_low)
    print("Decision value for 90th percentile:", decision_value_high)

    # Plot histogram and box plot
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(spending_per_customer_filtered, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Spending')
    plt.ylabel('Frequency')
    plt.title(f'Spending Distribution for {department}')

    plt.subplot(1, 2, 2)
    plt.boxplot(spending_per_customer_filtered, vert=False)
    plt.xlabel('Spending')
    plt.title(f'Boxplot for {department}')

    plt.tight_layout()
    plt.show()


# Read the CSV file, skipping the first row
df = pd.read_csv("supermarket_fixed.csv", sep=';', skiprows=1, header=None)

# Convert all values to integers, keeping decimals
df = df.apply(lambda x: pd.to_numeric(x))

# Extract transaction data (products) for each customer
transaction_data = df.iloc[:, 1:]

def calculate_department_spending(transaction_data):
    department_spending_list = []

    # Iterate over each row in the transaction data
    for idx, row in transaction_data.iterrows():
        customer_spending = [0] * 18  # Initialize spending for each department to 0
        dept_idx = 0  # Initialize department index
        # Iterate over each value in the row
        for val in row:
            if pd.notnull(val):
                if dept_idx % 3 == 0:  # Department number
                    dept = int(val)
                elif dept_idx % 3 == 2:  # Spending
                    if 1 <= dept <= 18:
                        customer_spending[dept - 1] += val
                dept_idx += 1
        department_spending_list.append(customer_spending)

    return pd.DataFrame(department_spending_list, columns=range(1, 19))


# Calculate department spending for each customer
department_spending_df = calculate_department_spending(transaction_data)

# Identify outliers for spending in each department
for department in range(1, 19):
    spending_data = department_spending_df[department]
    print(f"Department: {department}")
    identify_spending_outliers(spending_data, f"Department {department}")
