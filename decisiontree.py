import numpy as np


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # Index of the feature used to split
        self.threshold = threshold    # Threshold for the feature to split on
        self.left = left              # Left subtree (less than or equal to threshold)
        self.right = right            # Right subtree (greater than threshold)
        self.value = value            # Class label or prediction (for leaf nodes)

class DecisionTree:
    def __init__(self, max_depth = None):
        self.max_depth = max_depth 
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if len(np.unique(y)) == 1:
            return TreeNode(value=np.unique(y)[0]) # leaf node
        if self.max_depth is not None and depth >= self.max_depth:
            majority_class = np.bincount(y).argmax()
            return TreeNode(value=majority_class) # leaf node (max depth)

        # find the best split
        feature, threshold, left_y, right_y = self._best_split(X, y)
        if feature is None:
            majority_class = np.bincount(y).argmax()
            return TreeNode(value=majority_class) # leaf node (no split data)

        # split data
        left_X, right_X, _, _ = self._split_data(X, y, feature, threshold)
        left_node = self._build_tree(left_X, left_y, depth + 1)
        right_node = self._build_tree(right_X, right_y, depth + 1)

        return TreeNode(feature = feature, threshold = threshold, left = left_node, right = right_node)

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, node):
        if node.value is not None:
            return node.value
        if sample[node.feature] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)

    def _split_data(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _gini(self, y):
        classes = np.unique(y)
        impurity = 1
        for c in classes:
            prob_c = np.sum(y==c)/len(y)
            impurity -= prob_c ** 2
        return impurity

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold, best_left_y, best_right_y = None, None, None, None
        n_features = X.shape[1]
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_X, right_X, left_y, right_y = self._split_data(X, y, feature, threshold)
                if len(left_y) == 0 or len(right_y) == 0:
                    continue 
                gini = (len(left_y) / len(y) * self._gini(left_y) +
                        len(right_y) / len(y) * self._gini(right_y))
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature 
                    best_threshold = threshold
                    best_left_y = left_y
                    best_right_y = right_y 

        return best_feature, best_threshold, best_left_y, best_right_y





