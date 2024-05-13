#clusters based on for each customer divided by department (18) how many items they buy in one or 0
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
department_counts_df.dropna(inplace=True)

# Find the optimal number of clusters
find_optimal_clusters(department_counts_df, 10)

# Cluster the customers using KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
department_counts_df['cluster'] = kmeans.fit_predict(department_counts_df)

# Save the clustered data
department_counts_df.to_csv("clustered_customers_counts.csv", index=False)

# Print the first few rows of the clustered data
print("Clustered data:")
print(department_counts_df.head())

# Read the clustered data
clustered_data = pd.read_csv("clustered_customers_counts.csv")

# Extract the cluster labels and counts data
cluster_labels = clustered_data['cluster']
counts_data = clustered_data.drop(columns=['cluster'])

# Calculate the mean counts for each department in each cluster
cluster_means = counts_data.groupby(cluster_labels).mean()

# Plot the scatter plot
plt.figure(figsize=(12, 8))

# Define colors for each cluster
colors = ['blue', 'green', 'red', 'orange', 'purple']

# Plot scatter points for each department
for department in range(1, 19):
    department_counts = []
    for cluster in range(len(cluster_means.columns)):
        department_counts.append(cluster_means.loc[department][cluster])
    plt.scatter([department] * len(department_counts), department_counts, label=f'Department {department}', color=colors[:len(cluster_means.columns)], alpha=0.6)

plt.xlabel('Department')
plt.ylabel('Mean Product Counts')
plt.title('Mean Product Counts per Department for Each Cluster')
plt.xticks(range(1, 19), rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
