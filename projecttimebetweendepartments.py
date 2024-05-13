import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def identify_time_outliers(department_time_df, department=None):
    # Filter out customers who spent 0 time in the department
    time_per_customer_filtered = department_time_df[department_time_df > 0]

    # Calculate decision values
    decision_value_low = np.percentile(time_per_customer_filtered, 10)
    decision_value_high = np.percentile(time_per_customer_filtered, 90)

    # Identify outliers based on time thresholds
    outliers_low_time = np.where(time_per_customer_filtered < decision_value_low)[0]
    outliers_high_time = np.where(time_per_customer_filtered > decision_value_high)[0]

    print("Number of low time outliers:", len(outliers_low_time))
    print("Number of high time outliers:", len(outliers_high_time))
    print("Decision value for 10th percentile:", decision_value_low)
    print("Decision value for 90th percentile:", decision_value_high)

    # Plot histogram and box plot
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(time_per_customer_filtered, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.title(f'Time Distribution for {department}')

    plt.subplot(1, 2, 2)
    plt.boxplot(time_per_customer_filtered, vert=False)
    plt.xlabel('Time (seconds)')
    plt.title(f'Boxplot for {department}')

    plt.tight_layout()
    plt.show()


# Read the CSV file, skipping the first row
df = pd.read_csv("supermarket_fixed.csv", sep=';', skiprows=1, header=None)

# Convert all values to integers, keeping decimals
df = df.apply(lambda x: pd.to_numeric(x))

# Extract transaction data (products) for each customer
transaction_data = df.iloc[:, 1:]


def calculate_department_time(transaction_data):
    department_time_list = []

    # Iterate over each row in the transaction data
    for idx, row in transaction_data.iterrows():
        customer_time = [0] * 18  # Initialize time for each department to 0
        dept_idx = 0  # Initialize department index
        time_elapsed = 0  # Initialize time elapsed
        # Iterate over each value in the row
        for val in row:
            if pd.notnull(val):
                if dept_idx % 3 == 0:  # Department number
                    dept = int(val)
                elif dept_idx % 3 == 1:  # Time elapsed
                    time_elapsed += val
                elif dept_idx % 3 == 2:  # Spending
                    if 1 <= dept <= 18:
                        customer_time[dept - 1] += time_elapsed
                    time_elapsed = 0  # Reset time elapsed after recording spending
                dept_idx += 1
        department_time_list.append(customer_time)

    return pd.DataFrame(department_time_list, columns=range(1, 19))

# Calculate department time for each customer
department_time_df = calculate_department_time(transaction_data)

print(department_time_df.head())  # Print the first few rows to inspect the data


# Identify outliers for time in each department
for department in range(1, 19):
    time_data = department_time_df[department]
    print(f"Department: {department}")
    identify_time_outliers(time_data, f"Department {department}")
