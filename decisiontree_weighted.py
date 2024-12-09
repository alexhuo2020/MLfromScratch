import numpy as np


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # Index of the feature used to split
        self.threshold = threshold    # Threshold for the feature to split on
        self.left = left              # Left subtree (less than or equal to threshold)
        self.right = right            # Right subtree (greater than threshold)
        self.value = value            # Class label or prediction (for leaf nodes)


class DecisionTree:
    def __init__(self, max_depth = 3, is_classification = True):
        self.max_depth = max_depth 
        self.is_classification = is_classification
        self.tree = None
        

    def fit(self, X, y, class_weight):
        self.class_weight = np.ones(len(y)) / len(y) if class_weight is not None else np.ones_like(y)
        self.tree = self._build_tree(X, y, class_weight, depth=0)

    def _majority_class(self, y, weight):
        classes = np.unique(y)
        weighted_counts = {c: np.sum(weight[y == c]) for c in classes}
        return max(weighted_counts, key=weighted_counts.get)

    def _build_tree(self, X, y, weight, depth):
        n_samples, n_features = X.shape
        if len(np.unique(y)) == 1:
            return TreeNode(value=np.unique(y)[0]) # leaf node
        if self.max_depth is not None and depth >= self.max_depth:
            if self.is_classification:
                majority_class = self._majority_class(y, weight)
            else:
                majority_class = np.mean(y)
            return TreeNode(value=majority_class) # leaf node (max depth)

        # find the best split
        feature, threshold, left_y, right_y = self._best_split(X, y, weight)
        if feature is None:
            if self.is_classification:
                majority_class = self._majority_class(y, weight)
            else:
                majority_class = np.mean(y)
            return TreeNode(value=majority_class) # leaf node (no split data)

        # split data
        left_X, right_X, _, _, left_weight, right_weight = self._split_data(X, y, weight, feature, threshold)
        left_node = self._build_tree(left_X, left_y, left_weight, depth + 1)
        right_node = self._build_tree(right_X, right_y, right_weight, depth + 1)

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

    def _split_data(self, X, y, weight, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return X[left_mask], X[right_mask], y[left_mask], y[right_mask], weight[left_mask], weight[right_mask]

    def _gini(self, y, weight):
        classes = np.unique(y)
        impurity = 1
        total_weight = np.sum(weight)
        for c in classes:
            prob_c = np.sum(weight[y==c])/total_weight#len(y)
            impurity -= prob_c ** 2
        return impurity
    
    def _var(self, y):
        return np.var(y)

    def _best_split(self, X, y, weight):
        best_gini = float('inf')
        best_feature, best_threshold, best_left_y, best_right_y = None, None, None, None
        n_features = X.shape[1]
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_X, right_X, left_y, right_y, left_weight, right_weight = self._split_data(X, y, weight, feature, threshold)
                if len(left_y) == 0 or len(right_y) == 0:
                    continue 
                if self.is_classification:
                    gini = (np.sum(left_weight) /(np.sum(weight)) * self._gini(left_y, left_weight) + 
                            np.sum(right_weight) /(np.sum(weight)) *self._gini(right_y, right_weight))
                    # gini = (len(left_y) / len(y) * self._gini(left_y) +
                    #         len(right_y) / len(y) * self._gini(right_y))
                else:
                    gini = (np.sum(self.class_weight[left_y]) /np.sum(self.class_weight) * self._var(left_y) + 
                            np.sum(self.class_weight[right_y]) / np.sum(self.class_weight) *self._var(right_y)) # this seems not correct, to be corrected later

                    # gini = (len(left_y) / len(y) * self._var(left_y) +
                    #         len(right_y) / len(y) * self._var(right_y))
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature 
                    best_threshold = threshold
                    best_left_y = left_y
                    best_right_y = right_y 

        return best_feature, best_threshold, best_left_y, best_right_y




