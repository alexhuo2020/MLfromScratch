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
	+ $centroids$: k x `n_features`
	+ use `np.linalg.norm(X[:,np.newaxis] - centroids, axis=2)` to get all the distances, size `n_samples x k x n_features`
	+ use `labels=np.argmin(distances, axis=1)` to get the partition indices
- to get the mean, just call `X[labels==i].mean()` to get the mean of each cluster


## KNN
** The problem**

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

- Gini index: $1 - \sum_i p_i^2$
	+ $p_i$ is the probability of class $i$

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
