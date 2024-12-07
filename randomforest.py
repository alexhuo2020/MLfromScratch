import numpy as np 
from collections import Counter 
from decisiontree import DecisionTree

class RandomForest:
    def __init__(self, n_estimators = 100, max_features = 'sqrt', max_depth = None, bootstrap = True, max_samples = None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth 
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        k = self.max_samples if self.max_samples is not None else n_samples
        indices = np.random.choice(n_samples, k, replace = True)
        return X[indices], y[indices]
    
    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features 
        else:
            raise ValueError("Invalid value for max features")
        
    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[0]
        max_features = self._get_max_features(n_features)

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample[:max_features], y_sample[:max_features])
            self.trees.append(tree)
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds)
