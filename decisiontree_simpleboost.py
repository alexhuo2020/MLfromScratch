import numpy as np 
from decisiontree import DecisionTree

class SimpleBoosting:
    def __init__(self, n_estimators = 100, learning_rate = 0.1, max_depth = 3):
        self.n_estimators = n_estimators 
        self.learning_rate = learning_rate 
        self.max_depth = max_depth 
        self.trees = []
        self.initial_model = None ## for storing first const model
    
    def fit(self, X, y):
        self.initial_model = np.mean(y) 
        residual = y - self.initial_model 
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, is_classification=False)
            tree.fit(X, residual)
            self.trees.append(tree)

            # predict residuals to get the next tree
            predictions = tree.predict(X)
            residual -= self.learning_rate * predictions 
    
    def predict(self, X):
        predictions = np.full(X.shape[0], self.initial_model)
                # Add contributions from each tree
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions
