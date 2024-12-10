# Machine learning algorithm from scratch

## Linear Regression
**Problem** 

Given data set $\{(X_i,y_i)\}_{i=1}^N$, $X_i \in \mathbb{R}^{p}, y \in \mathbb R$, where $p$ is the dimension of features. Find the a line
$$y = X \beta +  \beta_0 $$
such that the mean squared error (MSE) is minimal. 
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N \left(y_i - (\beta_0 + \sum_{j=1}^p X_{ij} \beta_j\right)^2$$

**The algorithm**
There are two ways to solve the linear regression problem: gradient descent and use inverse matrix formula.
- gradient descent
	+ compute gradient of MSE with respect to $\beta$:
		$$\frac{\partial \text{MSE}}{\partial\beta_j} = \frac{2}{N} \sum_{i=1}^N  \left(y_i - (\beta_0 + \sum_{k=1}^p X_{ik} \beta_j\right) X_{ij} $$
		$$\frac{\partial \text{MSE}}{\partial\beta_0} = \frac{2}{N} \sum_{i=1}^N  \left(y_i - (\beta_0 + \sum_{k=1}^p X_{ik} \beta_j\right) $$
	+ gradient descent
		+ $\beta_j \gets \beta_j - \eta \frac{\partial \text{MSE}}{\partial\beta_j}$
	
- inverse matrix formula
	+ $\hat \beta = (X^T X)^{-1} X^T y$
	+ here we add $1$ and $X$ is given by
		$$X = \left(\begin{array}{ccc} 1 & x_{11} & \cdots & x_{1p} \\
										1 & x_{21} & \cdots & x_{2p} \\
										\vdots & \vdots & \ddots & \vdots \\
										1 & x_{N1} & \cdots & x_{Np}\end{array}\right)$$

**Code implementations**
- express input $X=(X_{ij}) \in \mathbb{R}^{N\times p}$ as matrix and the gradient can be computed by $1/N * np.dot(X.T, y_pred - y) $
- for using the formula directly, we need insert 1 to the first column. `np.c_[np.ones((X.shape[0], 1)), X]` (concatenate along second axis)



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
result = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds) # classification
result = np.mean(tree_preds) # regression
```

Why bootstrap is done with replacement?
- If we take the sample size equal the dataset, probability that a specific point is not included in a single sample is:
$$\left(1-\frac1n\right)^n \to e^{-1} = 0.368$$
about 36.8% data is left out in each bootstrap sample.
- If we do without replacement, then all data are used, trees train on identical datasets and will not benefit from bootstrap sampling.


### Boosting
Boosting aims to convert weak learners to strong learns.

#### Simple boosting
Let's build a series of decision trees, with the later one learn the residual of the previous one.

Let's take a regression problem as example. The objective function is 
$$L = \frac{1}{n} \sum_i (\hat y_i - y_i)^2$$
- $\hat y_i$: predictions
- $y_i$: true value

Let's take a look at a simple boosting process:
- Let's start with a simple model $G_0(x_i) = \frac{1}{N} \sum_{i=1}^N y_i$, which is a constant.
- compute residual $$r_0(x_i) = y_i - G_0(x_i)$$
- use a model (e.g. decision tree) on the residual, i.e. fit $r_0(x_i)$ with $G_1(x_i)$.
- update the residual $$r_1(x_i) = y_i - G_0(x_i) - \eta G_1(x_i)$$
- do the above steps iteratively, using a new model at each step
	+ $r_s(x_i) \gets r_{s-1}(x_i) - \eta G_{s-1} (x_i)$
	+ fit $\{(x_i,r_s(x_i))\}_{i=1}^N$ with model $G_s$
- final model is $$G(x) = \sum_{s=1}^m \eta G_s(x) + G_0$$

Why a learning rate is used?
- $G_s(x_i) = (r_{s+1}(x_i) - r_s(x_i)) /\eta$, since the residuals are usually small, dividing a learning parameters will make it bigger and easier to learn.
- this is actually gradient descent method (will explain later)

**Code Implementation**
- The first model ($G_0$) is a constant `initial_model = np.mean(y)`
- compute residual `residual = y - initial_model`
- fit residual `tree.fit(residual)`
- after fit, compute $G_s$ for the next residual `residual -= learning_rate * tree.prediction(X)`
- to make predictions add all tree predictions together `initial_model + learning_rate * sum([tree.prediction(X) for tree in trees])`

### Simple boosting for binary classification problems

For classification problem, the residual is not that simple. For binary classification problem, we start with the log-odds, given by 
$$F_0(x) = \log \frac{\bar p}{1 - \bar p}$$
- $\bar p$ is the mean of y.
To compute residuals, we use the regression tree output instead of the classification tree, so that the output is a float number instead of a class label. We transform the predictions with a sigmoid function 
- residuals $$r_s(x_i) = y_i - \hat p_i,\quad \hat p_i = \frac{1}{1 + e^{-F_s(x_i)}}$$
	+ the output of the decision tree is a float number and we use sigmoid to convert it to probability
	+ to make predictions, one can set a threshould, say 0.5 
	+ train model $G_s$ to fit $\{(x_i,r_s(x_i))\}$
- update predictions by 
$$F_{s} \gets  F_{s-1}  + \eta G_{s-1}$$
- final predictions given by $$\hat p_S = \frac{1}{1+e^{-F(s)}}$$ and label given by $\hat p_S \ge 0.5$.

**differences with regression**
- we cannot simply add the predictions of the trees, since the residual relationship is not linear. 
- instead, we compute the updated residuals each iteration


### Gradient Boosting in General
**Gradient Boosting**
- both for classification and regression
- builds a predictive model in a stage-wise fashion by sequentially adding weak learners (typically decision trees) to minimize a loss function
- **gradient** refers to use gradient descent to optimize the loss function

**Key Concepts**
- Weak Learner: a model that performs slightly better than random guessing, such as a shallow decision tree
- Ensemble: final model is an ensemble of all the weak learners added in a step-by-step manner
- Additive training: each model correct the errors of the combined ensemble so far
- gradient descent: minimize the loss function by descending the gradient

**Math Theory**
- learning objective (loss function) $L(y,F(x))$
	+ $F(x)$ is the model prediction, in regression, $L = \frac{1}{N}\sum_i (y_i - F(x_i))^2$ 
	+ in classification, the loss is more complicated, in the above example, the loss is the **Binary Cross-Entropy (BCE) Loss**
	$$L = -\frac{1}{N} \sum_{i=1}^N \left[y_i \log \hat p_i + (1-y_i) \log (1-\hat p_i)\right]$$
	$$\hat p_i = \frac{1}{1+e^{-F(x_i)}}$$

- gradient descent is to update $F$ by $$F \gets F - \eta \frac{\partial L}{\partial F}$$
	+ in regression $\frac{\partial L}{\partial F} = \frac{2}{N} (y_i - F(x_i)) =  (2/N)r$, where $r$ is the residual
	+ in classification $\frac{\partial L}{\partial F} = y_i - \hat p_i  =r$
	+ $F$ is a function so the derivative is actually a functional derivative (ignore this if you donot understand)

- in the $k$-th step, we need to minimizing $L(y, F_k(x))$. 
	+ fit the gradient $\partial L /\partial F(y,F_{k-1}(x))$ with $G_k$
	+ and add it to the ensemble $F_k = F_{k-1} + \eta G_k$

- the learining rate has an optimal value, which can be solved by 
	$$\eta_k=\min_{\eta} L(y,F_{k-1}(x) + \eta G_k)$$
	+ we can use this to do the update $F_k = F_{k-1} + \eta_k G_k$
	+ without an optimal learning rate, algorithm still works but just maynot be optimal

- Final ensemble of models $$F_S = \sum_{k=1}^S \eta_k G_k + G_0$$


### AdaBoost
Adaptive method for gradient descent is that the learning rate is adjusted by a weight considering history of the gradients. The **adaptive** property is 
- subsequent weak learners (models) are adjusted in favor of instances misclassified by previous models

AdaBoost
- primarily for bindary classification 


**Derivation of the algorithm**
- The loss function
	$$L(y,F(x)) = \sum_{i=1}^N e^{-y_i F(x_i)}$$
	+ $y_i \in \{-1,1\}$
	+ $F = \sum_{k=1}^K \alpha_k G_k(x)$

- gradient descent
	+ comptue the gradient
	$$\frac{\partial L}{\partial F}(x_i) = - y_i e^{-y_i F_{k-1}(x_i)}$$
	+ fit the gradient with $G_k$
	+ do gradient descent:
	$$F_k(x_i) = F_{k-1} (x_i) - \eta \frac{\partial L}{\partial F}(x_i) = F_{k-1}(x_i) + \eta G_k(x_i) $$

- update the learning rate
	+ we then update the learning rate by minimizing
		$$\eta_k=\min_{\eta} L(y,F_{k-1}(x) + \eta G_k)$$

	+ let's do this by solving $$\frac{\partial L(y, F_{k-1} + \eta G_k)}{\partial \eta} = 0$$

		+ Rewrite the loss function as $$L_k = \sum_{i} e^{-y_i \left(F_{k-1}(x_i) + \eta G_k(x_i)\right)} = \sum_{i} e^{-y_i F_{k-1}(x_i)} e^{-\eta y_i G_k(x_i)}:=\sum_{i=1}^N w_i^k e^{-\eta y_i G_k(x_i)}$$
		+ the weight $w_i^k:= e^{-y_i G_{k-1}(x_i)}$ is carried forward from previous iterations
		+ $L_k$ can be splitted to correct and incorrect classifications by $G_k$
		$$L_k = \sum_{i: G_k(x_i) = y_i} w_i^k e^{-\eta} +  \sum_{i: G_k(x_i) \not= y_i} w_i^k e^{-\eta}$$
		+ Rewrite in the form 
		$$L_k = (1-\varepsilon_k) e^{-\eta} + \varepsilon_k e^{\eta}$$
		$$\varepsilon_k = \frac{\sum_{i: G_k(x_i) \not= y_i} w_i^k }{\sum_i w_i^k}$$
		+ compute the gradient $\partial L / \partial \eta$:
		$$\frac{\partial L}{\partial \eta} = -\eta (1-\varepsilon_k) e^{-\eta} + \varepsilon_k \eta e^{\eta}=0 $$
		+ solve it to get $$\eta_k=\frac12\log \frac{1-\varepsilon_k}{\varepsilon_k} $$

- the final update rule is 
	$$F_k(x_i) =  F_{k-1}(x_i) + \eta_k G_k(x_i),\quad \eta_k = \frac12\log \frac{1-\varepsilon_k}{\varepsilon_k} $$
	+ since $\varepsilon_k$ depends on $w_k$, we can also derive the update rule for $w_k$ using its definition:
	$$w_i^k = w_i^{k-1} e^{-\eta_k y_i G_k(x_i)} = \left\{\begin{array}{ccc} w_i^{k-1} e^{-\eta_k} & \text{ if }y_i = G_k(x_i) \\  w_i^{k-1} e^{\eta_k} & \text{ if }y_i \not= G_k(x_i)\end{array}\right.$$
	+ $w_i$ is usually normalized $w_i^k = w_i^k/ \sum_{j=1}^N w_j^{k}$ before passing to the next iteration.

- final strong classifier 
	$$F(x) = \text{sign} \left(\sum_{i=1}^K \eta_k G_k(x)\right)$$

**The algorithm**
- Initialize $w_i^0=\frac{1}{N}$ for all $i=1,\ldots,N$.
- for  k from 1 to K: 
	+ train week decision tree $G_k$ to fit $\sum_{i=1}^N 1_{y_i \not= F_{k-1} (x_i)} e^{-y_i F_{k-1}(x_i)} = \sum_{i=1}^N 1_{y_i \not= F_{k-1} (x_i)} w_i^{k-1} = \varepsilon_{k-1}$ (since the model $F_{k-1}$ already fit some data well and we want to keep it)
	+ compute $\eta_k = \frac12 \log \frac{1-\varepsilon_k}{\varepsilon_k}$
	+ update weights $w_i^k = w_i^{k-1} e^{-\eta_k y_i G_k(x_i)}$
- output strong classifier
	$$F(x) = \text{sign} \left(\sum_{i=1}^K \eta_k G_k(x)\right)$$


Note that we need a decision tree supporting weights, this can be done by updating the gini computation by 
$$\text{Gini}(p) = 1-\sum_k p_k^2,\quad p_k = \frac{\sum_{i\in S_k} w_i}{\sum_i w_i} $$
and also the computation of gini impurity for the splits.
$$\text{impurity} = \frac{\sum_{i\in S_{\text{left}}} w_i}{\sum_{i\in S} w_i} \text{impurity}_{\text{left}} + \frac{\sum_{i\in S_{\text{right}}} w_i}{\sum_{i\in S} w_i} \text{impurity}_{\text{right}}$$

