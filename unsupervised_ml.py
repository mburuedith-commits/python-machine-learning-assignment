import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter

# Prepare customer data (Monthly Spending KES, Monthly Visits)
customer_data = [
    [1000, 2],
    [1500, 3],
    [2000, 2],
    [5000, 6],
    [6000, 7],
    [5500, 5],
    [12000, 12],
    [15000, 14],
    [13000, 11],
    [3000, 4]
]

# Scale features using StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data)

# Create and train K-Means model (n_clusters=3, random_state=42, n_init=10)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_data)

# Display the cluster assigned to each customer
for index, label in enumerate(cluster_labels):
    print(f"Customer {index + 1} belongs to Cluster {label}")

# Display total number of customers in each cluster
print("\nCluster Summary Counting:")
counts = Counter(cluster_labels)
for cluster_id, count in sorted(counts.items()):
    print(f"Cluster {cluster_id}: {count} customers")

# --- BONUS TASK: Matplotlib Scatter Plot ---
print("\nGenerating customer segments graph...")
spending = [row[0] for row in customer_data]
visits = [row[1] for row in customer_data]

plt.figure(figsize=(8, 6))
scatter = plt.scatter(spending, visits, c=cluster_labels, cmap='viridis', s=100, edgecolors='black')
plt.title('Customer Segmentation via K-Means Clustering')
plt.xlabel('Monthly Spending (KES)')
plt.ylabel('Monthly Visits')
plt.grid(True)

# Add legend matching colors to cluster indices
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()
