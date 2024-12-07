import numpy as np
from collections import Counter
class kNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors 
        self.X = None
        self.y = None
  
    def fit(self, X, y):
        self.X = X
        self.y = y
  
    def predict(self, X):
        predictions = [self._predict_single(xx) for xx in X]
        return np.array(predictions)

    def _predict_single(self, x):
        distances = np.linalg.norm(self.X - x, axis=1)
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = self.y[k_indices]

        if np.issubdtype(self.y.dtype, np.integer):
            return Counter(k_nearest_labels).most_common(1)[0][0]
        else:
            return np.mean(k_nearest_labels)



