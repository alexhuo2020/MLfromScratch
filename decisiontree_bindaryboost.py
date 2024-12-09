import numpy as np 
from decisiontree import DecisionTree

class BinaryBoosting:
    def __init__(self, n_estimators = 100, learning_rate = 0.1, max_depth = 3):
        self.n_estimators = n_estimators 
        self.learning_rate = learning_rate 
        self.max_depth = max_depth 
        self.trees = []
        self.initial_model = None ## for storing first const model
    
    def fit(self, X, y):
        pbar = np.mean(y)
        epsilon = 1e-10
        self.initial_prediction = np.log((pbar + epsilon) / (1 - pbar + epsilon))
        F_m = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            phat = 1 / (1+np.exp(-F_m))
            residual = y - phat
            tree = DecisionTree(max_depth=self.max_depth, is_classification=False)
            tree.fit(X, residual)
            self.trees.append(tree)

            # predict residuals to get the next tree
            predictions = tree.predict(X)
            F_m += self.learning_rate * predictions
    
    def predict_prob(self, X):
        F_m = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            F_m += self.learning_rate * tree.predict(X)
        probabilities = 1 / (1+np.exp(-F_m))
        return probabilities
    
    def predict(self, X):
        probabilities = self.predict_prob(X)
        return (probabilities >= 0.5).astype(int)
