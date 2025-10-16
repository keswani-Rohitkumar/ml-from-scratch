import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters = 100, tol = 1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self,X):
        random_indices = np.random.choice(len(X), self.k,replace= False)
        self.centroids = X[random_indices]

        for iteration in range(self.max_iters):

            labels = self._assign_cluster(X)

            new_centroids = np.array(
                [
                    X[labels == i].mean(axis=0) if len(X[labels == i]) >0 else self.centroids[i] for i in range(self.k)
                ]
            )

            shift = np.linalg.norm(self.centroids - new_centroids)
            if shift < self.tol:
                print(f"Converged in {iteration+1} iterations.")
                break
            
            self.centroids = new_centroids
        self.labels_ = labels

    def _assign_cluster(self,X):

        distance = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

        return np.argmin(distance, axis=1)
    
    def predict(self, X):
        return self._assign_cluster(X)
        
# Generate synthetic data
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)

# Apply custom K-Means
kmeans = KMeans(k=3, max_iters=100)
kmeans.fit(X)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=40, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], color='red', marker='x', s=200)
plt.title("K-Means Clustering (From Scratch)")
plt.show()
