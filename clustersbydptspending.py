#clusters based on for each customer divided by department (18) how much they spend in one or 0

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read the CSV file, skipping the first row
df = pd.read_csv("supermarket_fixed.csv", sep=';', skiprows=1, header=None)

# Convert all values to integers, keeping decimals
df = df.apply(lambda x: pd.to_numeric(x))

# Extract customer index from the first column
customer_index = df.iloc[:, 0]

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


# Determine the optimal number of clusters using the elbow method
def find_optimal_clusters(data, max_k):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), inertias, marker='o', linestyle='-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()


# Drop any rows with missing values
department_spending_df.dropna(inplace=True)

# Find the optimal number of clusters
find_optimal_clusters(department_spending_df, 10)

# Cluster the customers using KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
department_spending_df['cluster'] = kmeans.fit_predict(department_spending_df)

# Save the clustered data
department_spending_df.to_csv("clustered_customers.csv", index=False)

# Print the first few rows of the clustered data
print("Clustered data:")
print(department_spending_df.head())

# Read the clustered data
clustered_data = pd.read_csv("clustered_customers.csv")

# Extract the cluster labels and spending data
cluster_labels = clustered_data['cluster']
spending_data = clustered_data.drop(columns=['cluster'])

# Calculate the mean spending for each department in each cluster
cluster_means = spending_data.groupby(cluster_labels).mean()

# Transpose the DataFrame for easier plotting
cluster_means = cluster_means.transpose()

# Normalize the data for each cluster
cluster_means_normalized = cluster_means.div(cluster_means.sum(axis=0), axis=1)

# Plot the bar chart
plt.figure(figsize=(12, 8))

# Define colors for each cluster
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Plot bars for each cluster
for i, cluster in enumerate(cluster_means_normalized.columns):
    plt.bar(cluster_means_normalized.index, cluster_means_normalized[cluster], label=f'Cluster {cluster}', color=colors[i], alpha=0.6)

plt.xlabel('Department')
plt.ylabel('Normalized Mean Spending')
plt.title('Normalized Mean Spending per Department for Each Cluster')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
