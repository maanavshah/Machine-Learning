# K-Means Clustering

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the Elbow method to find optimal number of clusters
max_clusters = 10
wcss = []
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, max_clusters + 1), wcss)
plt.title('Elbow method (Optimal centroids)')
plt.xlabel('Number of centroids')
plt.ylabel('WCSS')
plt.show()

# Applying KMeans with optimal number of centroid
kmeans = KMeans(n_clusters=5, init='k-means++',
                max_iter=300, n_init=10, random_state=0)
y_predict = kmeans.fit_predict(X)

# Visualising the cluster
plt.scatter(X[y_predict == 0, 0], X[y_predict == 0, 1],
            s=100, c='red', label='Cluster 1')
plt.scatter(X[y_predict == 1, 0], X[y_predict == 1, 1],
            s=100, c='green', label='Cluster 2')
plt.scatter(X[y_predict == 2, 0], X[y_predict == 2, 1],
            s=100, c='blue', label='Cluster 3')
plt.scatter(X[y_predict == 3, 0], X[y_predict == 3, 1],
            s=100, c='magenta', label='Cluster 4')
plt.scatter(X[y_predict == 4, 0], X[y_predict == 4, 1],
            s=100, c='cyan', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=200, c='yellow', label='Centroid')
plt.title('KMeans')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
