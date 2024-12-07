# Machine learning algorithm from scratch

## KMeans
**The problem**

Given a set of observations $(X_1, \ldots, X_n)$, find a partition into k sets $S = \{S_1,\ldots,S_k\}$, that minimizes the distance to clusters' means
$$\arg\min_{S} \sum_{i=1}^k \sum_{x\in S_i} \|x - \mu_i\|^2$$
- $\mu_i = \frac{1}{|S_i|} \sum_{x\in S_i} x$

**The algorithm:**

- initialize centroids by randomly take $k$ points from the observations
- for each data points, assign a centroid, get the partition $S_i$
- find the mean of points in a cluster $\mu_i$
- repeat until centroids converges $\mu_i^s \approx \mu_i^{s+1}$ or max iterations is reached.

Code implementation guides
- use `np.random.choices(n_samples, n_clusters, replace=False)` for the first step
- to get the partition, we just compute the distances from a point to the centroids and find the smallest one.
	+ $X$: `n_samples` x `n_features`
	+ centroids: k x `n_features`
	+ use `np.linalg.norm(X[:,np.newaxis] - centroids, axis=2)` to get all the distances, size `n_samples x k x n_features`
	+ use `labels=np.argmin(distances, axis=1)` to get the partition indices
- to get the mean, just call `X[labels==i].mean()` to get the mean of each cluster


## KNN
**The problem**

Given observations $(X_i, y_i), i=1,\ldots, n$. For a new point $x$, find the $k$ nearest neighbors, i.e.
$$\|X_{(1)} - x\| \le \cdots \le \|X_{(k)} - x\|$$
then 
- for classification problem, find the most common class in $y_{(i)}, i=1,\ldots,k$
- for regression problem, find the mean of $y_{(i)}$

Code implementaion guides:
- use `np.argsort(distances)[:k]` to get the nearest neighbors
- use `Counter(y[nearest indices]).most_common(1)[0][0]` to get the most common class in y
- use `np.issubdtype(y.dtype, np.integer)` to choose between classification and regression

## Decision Trees
Decision tree is a **supervised learning** method both for **classification** and **regression**.

Given observations $(X_i, y_i), i=1,\ldots, n$. A decision tree makes decision based on thresholds set on features. 

- Gini impurity: $1 - \sum_i p_i^2$
	+ $p_i$ is the probability of class $i$
	+ the smaller, the more pure
	+ in decision tree, we want to split the data in such a way that the resulting child nodes are as pure as possible

Example:
- (1,0), (2,0), (3,1)
- if threshold is taken to be 1, then 
	+ left (1,0), gini impurity = 1-1=0
	+ right (2,0) (3,1) gini impurity = $1 - 0.5^2*2 = 0.5$
	+ total gini impurity = 1/3 * 0 + 2/3 * 0.5 = 0.33
- if threshold = 2
	+ left (1,0) (2,0), gini impurity = 1 - 1 = 0
	+ right (3,1) gini impurity = 0
- if threshold = 3
	+ gini impurity = 1 - (2/3)^2 - (1/3)^2 = 0.45
- so the smallest gini corresponds to threshold = 2


The function to compute gini impurity
```python
def gini(y):
	classes = np.unique(y)
	impurity = 1
	for c in classes:
		prob_c = np.sum(y==c)/len(y)
		impurity -= prob_c ** 2
	return impurity
```

Code implementations
- the final decisions are leaf nodes, only has values (classes)
- the root: feature, threshold, left node, right node
- left node: <= threshold, right node: > threshold
- to find the best split
	+ for each feature and threshold, compute the gini impurity, and find the best feature and threshold
	+ split into left and right; for left and right, find the best split and do it iteratively
- use bfs to build the tree
	+ `left_node = build_tree(left_X, left_y, depth+1)`
	+ `right_node = build_tree(right_X, right_y, depth+1)`
	+ return the root, `TreeNode(best_feature, best_threshold, left_y, right_y)`
	+ Three cases for the leaf node
		+ reach max depth
		+ no split data (feature=None)
	 	+ only 1 class in y
- to predict, search algorithm in binary search tree
	+ `if x[node.feature] <= node.threshold: return _predict(x, node.left)`
	+ `if x[node.feature] > node.threshold: return _predict(x, node.right)`

### Decision trees for regression
For decision trees for regression, the gini index will not be used since it only applied to discrete random variables. Instead, we use variance. 
$$\text{Var}(y) = \frac{1}{N}\sum_{i=1}^N (y_i - \bar y)^2$$
- we want to make the node closer to the mean, so we use variance
- to get the leaf node value, we use averaging instead of majority voting

Based on the above observations, we need to change the code `gini` to `var` and `np.bincount(y).argmax()` to `y.mean()`. 


 
## Bias vs Variance
Bias and Variance are two fundamental sources of error in machine learning models, and they represent the tradeoff between underfitting and overfitting.

### Bias
- Definition: Bias is the error introduced by approximating a real-world problem (which may be complex) by a simplified model.
- Cause: High bias occurs when a model is too simple to capture the underlying patterns in the data. This is often the result of overly strong assumptions made by the model.
- Effect: A high-bias model systematically makes errors on the training data because it cannot capture the complexity of the relationships in the data. This leads to underfitting—where the model does not perform well on both the training set and the test set.
- Example: A linear regression model trying to fit a nonlinear relationship in the data. The linear model is too simple to capture the curve, leading to high bias.

**Signs of high bias:**

- The model performs poorly on both the training data and the test data.
- The model makes systematic errors, as it is unable to capture key patterns or relationships in the data.

### Variance
- Definition: Variance is the error introduced by the model’s sensitivity to small fluctuations or noise in the training data.
- Cause: High variance occurs when a model is too complex and overfits the training data. The model learns not only the underlying patterns but also the noise or random fluctuations in the data.
- Effect: A high-variance model performs well on the training data but poorly on the test data, as it fails to generalize. This leads to overfitting—where the model is too tailored to the training set and doesn’t work well with new, unseen data.
- Example: A decision tree with very deep branches that perfectly fits the training data but fails to generalize to new data because it captures too many details (including noise).

**Signs of high variance:**

- The model performs well on the training data but poorly on the test data.
- The model's predictions are highly sensitive to small changes in the input data.

### Tradeoff
- **Underfitting:** This occurs when the model has **high bias** and **low variance**. The model is too simple, making it unable to capture the complexity of the data.
- **Overfitting:** This occurs when the model has **low bias** and **high variance**. The model is too complex, capturing noise in the training data and failing to generalize well to unseen data.

**Reducing Bias:**
- Increase model complexity (e.g., use more features, choose more complex models).
- Use more sophisticated algorithms (e.g., gradient boosting, neural networks).

**Reducing Variance:**
- Use simpler models (e.g., linear regression, shallow decision trees).
- Use regularization techniques (e.g., L2 regularization, pruning in decision trees).
- Use ensemble methods (e.g., bagging, random forests) to average out the predictions.


For improvement over decision trees, we can apply bagging (random forests) or gradient boosting to reduce the variance and bias. 

### Bagging and Random Forests
**Bagging**: an ensemble method to reduce variance of the machine learning model, combining multiple model predictions together. Reduce overfitting.

**Steps**
1. bootstrap sampling: random create multiple subsets of sampling data **with replacement**.
2. Train separate model on each boostrap sample. For decision trees (random forests), we use a random subset of features for each model.
3. aggregate predictions
	+ classification: majority voting
	+ regression: averaging


Code implement
- for bootstrap sampling `np.random.choice(n_samples, max_samples, replace = True)`
- use the first `max_feature = sqrt(n_features)` of orther max feature functions
- for each sampled data, run decision tree, and aggregate the results `trees.append(tree)`
- for classification, majority voting 
```python
tree_preds = np.array([tree.predict(X) for tree in self.trees])
result = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds)
```
