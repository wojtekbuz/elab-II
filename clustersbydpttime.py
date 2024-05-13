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




# Find the optimal number of clusters
find_optimal_clusters(department_time_df, 10)

# Cluster the customers using KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
department_time_df['cluster'] = kmeans.fit_predict(department_time_df)

# Save the clustered data
department_time_df.to_csv("clustered_customers_time.csv", index=False)

# Print the first few rows of the clustered data
print("Clustered data:")
print(department_time_df.head())

# Read the clustered data
clustered_data = pd.read_csv("clustered_customers_time.csv")

# Extract the cluster labels and time data
cluster_labels = clustered_data['cluster']
time_data = clustered_data.drop(columns=['cluster'])

# Calculate the mean time for each department in each cluster
cluster_means = time_data.groupby(cluster_labels).mean()

# Transpose the DataFrame for easier plotting
cluster_means = cluster_means.transpose()

# Normalize the data for each cluster
cluster_means_normalized = cluster_means.div(cluster_means.sum(axis=0), axis=1)

# Plot the scatter plot
plt.figure(figsize=(12, 8))

# Define colors for each cluster
colors = ['blue', 'green', 'red', 'orange', 'purple']

# Plot scatter points for each department
for cluster in range(len(cluster_means.columns)):
    plt.scatter(range(1, 19), cluster_means.iloc[:, cluster], label=f'Cluster {cluster + 1}', color=colors[cluster], alpha=0.6)

plt.xlabel('Department')
plt.ylabel('Mean Time Spent (seconds)')
plt.title('Mean Time Spent per Department for Each Cluster')
plt.xticks(range(1, 19))
plt.legend()
plt.tight_layout()
plt.show()
