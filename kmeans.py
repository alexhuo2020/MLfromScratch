import numpy as np

class kMeans:
    def __init__(self, n_clusters, max_iters=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        for i in range(self.max_iters):
            self.labels = self._assign_clusters(X)
            new_centroids = np.array([X[self.labels==k].mean(axis=0) for k in range(self.n_clusters)])
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids
    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis = 1)

    def predict(self, X):
        return self._assign_clusters(X)


        
