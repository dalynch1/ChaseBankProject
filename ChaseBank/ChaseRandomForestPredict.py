import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('branch_consolidation_results.csv')

# Features for clustering
features = ['Latitude', 'Longitude', 'recent_avg_deposits']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Determine k using elbow method
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for KMeans')
plt.show()

# Fit KMeans with chosen 2
k = 2
kmeans = KMeans(n_clusters=k, random_state=1)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Inspect clusters
print('Cluster Sizes',df['Cluster'].value_counts())
print('Clusters Description',df.groupby('Cluster')[['recent_avg_deposits', 'Branches_Served', 'Deposits_Served']].describe())
print('Count by County',df.groupby(['Cluster', 'State'])['County'].value_counts())
print('Average deposits per cluster:',df.groupby('Cluster')['recent_avg_deposits'].mean())
print('Total deposits served per cluster:\n',df.groupby('Cluster')['Deposits_Served'].sum())
