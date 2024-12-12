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


**Metrics**
- MSE: Mean Squared Error, penalizes large errors more than smaller errors because of the squaring
- MAE: Mean Absolute Error, less sensitive to outlier than MSE
- $R^2$ score: $$1-\frac{\sum (y_i-\hat y_i)^2}{\sum (y_i - \bar y)^2}$$
	+ measures the proportion of variance in the dependent variable explained by the independent variables
	+ Closer to 1 indicates a better fit 
- adjusted $R^2$ score: $$1 - (1-R^2) \frac{N-1}{N-p-1}$$
	+ $p$ is the number of feature
	+ adjusts $R^2$ to penalize adding more features that do not improve the model significantly.



## Logistic Regression and Softmax Regression
For classification problems, the linear regression cannot be used. We need a different loss function.

**Entropy**

The entropy in information theory is defined as 
$$H(X) = -\sum_{x \in \mathcal X} p(x) \log p(x) =  \mathbb E [-\log p(X)] $$
- for binomial distribution, the emprical entropy on a observation $\{y_i\}_{i=1}^N$ equals $$H(X) = \frac1N \sum_{i=1}^N - \log p(y_i) $$

**Cross-Entropy**
To compare distribution $P$ and $Q$, we define the a distance $H(P,Q):=- E_p [\log q]$ and the minimal of this value occurs when $p=q$ due to the convexity of entropy.
- for discrete distributions, the cross entropy for  a sample equals $$H(P,Q) = - \frac1N \sum_{i=1}^N \log q(y_i)$$
	+ $p$ is the true distribution of $y_i$
	+ $q$ is another distribution
	+ by minimizing it, we can get $q \approx p$


**Logistic Regression**
- $q(y_i=1) = \frac{1}{1 + e^{- (X\beta + \beta_0)_i}}$
- binary distribution $y_i = 1 or 0$. Binary Cross Entropy (BCE):
	$$H(P,Q) = - \frac1N \sum_{i=1, y_i=1}^N  \log q_i -  \frac1N \sum_{i=1, y_i=0}^N  \log q_i 	$$
	which can be rewrite as 
	$$H(P,Q) = -\frac1N \sum_{i=1}^N \left(y_i \log q_i + (1-y_i) \log (1-q_i)\right)$$
- the gradient 
	$$\frac{\partial H}{\partial q_i} = -\frac{1}{N} \left(\frac{y_i}{q_i} - \frac{1-y_i}{1-q_i} \right)$$
	$$\frac{\partial H}{\partial \beta_j} = \sum_i \frac{\partial H}{\partial q_i} \frac{\partial q_i}{\partial \beta_j} =\sum_i  -\frac{1}{N} \left(\frac{y_i}{q_i} - \frac{1-y_i}{1-q_i} \right) (1-q_i) q_i (-X_{ij}) = -\frac1N \sum_i (y_i-q_i) X_{ij} $$
	$$\frac{\partial H}{\partial \beta_0} = -\frac1N \sum_i (y_i-q_i)$$
	+ the derivative is similar to that of the linear regression, except  that $q_i$ is replaced for the linear prediction

**Code Implementation**
- add `sigmoid` after the linear output in the linear regression code
- also add `sigmoid` in the predict function
- for sigmoid function, add a small number like $1e-15$ to avoid $log(0)$

**Softmax Regression**
- For classification with more than two classes, we can no longer use $p_i$ and $1-p_i$ for the computation. However, we should use the softmax function
	$$q(y_i = k) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, z_k = X \beta^k+\beta_0^k \beta$$
	+ for binary $k=0,1$ and it reduce to the sigmoid function
	+ we only need to change the sigmoid to softmax for multi-class classification
- Now the loss is the cross entropy loss 
	$$H(p,q) = - \frac1N \sum_{i=1}^N \sum_{k=1}^K y_{ik}\log q(y_i = k) $$

**Code Implementation**
- remember to initialize $\beta \in \mathbb{R}^{PxK}$, P is the number of features and K is the number of classes
- final prediction should convert to class label `np.argmax(y_pred, axis=1)`
- use a one hot encoding on $y_i$ before compute $\hat q -\text{OneHot}(y_i)$



**Metrics in classification**
Four cases
- True Positive
- False Positive
- True Negative 
- False Negative 

Metrics
- Accuracy: ratio of correctly predicted instances $$\text{Accuracy} = \frac{TP + TN}{All}$$
	+ Best for: Balanced datasets where all classes have roughly equal representation.
	+ Limitations: Misleading for imbalanced datasets.

- Precision (positive predictive value): proportion of positive predictions that are actually correct $\text{Precision} = \frac{TP}{TP + FP}$
	+ Best for: Situations where false positives are costly (e.g., spam detection)

- Recall (sensitivity): proportion of actual positives that are correctly identified $$\text{Recall} = \frac{TP}{TP + FN}$$
	+ Best for: Situations where false negatives are costly (e.g., medical diagnosis)

- F1 Score: harmonic mean of precision and recall $$\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
	+ Best for: Imbalanced datasets where precision and recall are both important.

**ROC (Receiver Operating Characteristic) Curve**
- shows trade-off between the True Positive Rate (TPR) (=Recall) and the False Positive Rate (FPR) at various classification thresholds.
	$$\text{FPR} = \frac{FP}{FP + TN}$$
	+ ROC curve evaluates the performance at all possible thresholds for classification
	+ interpret:
		+ Diagonal Line: Represents random guessing (e.g., TPR=FPR).
		+ Above Diagonal: Indicates better performance (more true positives than false positives).
		+ Closer to Top Left: Indicates optimal performance, with high TPR and low FPR.

**AUC (Area Under the Curve)**
- A single scalar value that quantifies the overall performance of the model by calculating the area under the ROC curve.
- Ranges:
	+ 1.0: Perfect classifier (all predictions are correct)
	+ 0.5: Random guessing (the ROC curve is a diagonal line)
	+ <0.5: Worse than random guessing (rare and indicates possible model issues)
- Interpret: The AUC score represents the probability that the model ranks a randomly chosen positive instance higher than a randomly chosen negative instance.
- ROC and AUC: Threshold Independence and Imbalance Robustness
- For highly imbalanced datasets (e.g., fraud detection where positives are rare): use Precision-Recall Curve instead because it focuses more on the performance related to the positive class.

Example:
- suppose a binary classifier for spam email detection
	+ TPR increases as we lower the threshold (catch more spam but risk false positives).
	+ FPR increases as we lower the threshold (more non-spam emails incorrectly classified as spam).

**Precision-Recall Curve**
- More insightful for imbalanced datasets because it emphasizes the performance on the minority (positive) class.
- Ideal for problems where identifying positives (e.g., fraud, rare disease) is critical.


Example:
1. Use Case: Fraud Detection
- Scenario: Detect fraudulent transactions in a banking system.

- Challenge: Fraud is rare (imbalanced dataset), and missing a fraudulent transaction (false negative) is costly.

- Metric Priority: High recall (to catch as many frauds as possible), even at the cost of reduced precision.

- Threshold Selection:
	+ Lower the threshold to increase recall.
	+ Example: Predict "fraud" if probability > 0.3 instead of 0.5.
	+ Trade-Off: More false positives (non-fraud marked as fraud), which can be reviewed manually.


2. Use Case: Medical Diagnosis
- Scenario: Predict the presence of a disease.

- Challenge: Missing a disease diagnosis (false negative) is critical, while a false positive might lead to unnecessary tests but no serious harm.

- Metric Priority: Maximize recall to minimize false negatives.

- Threshold Selection:
	+ Lower the threshold to ensure more positive predictions are flagged.
	+ Example: For a cancer detection model, classify as "positive" if the probability > 0.2.
	+ Trade-Off: More false positives, but fewer missed cases.

3. Use Case: Spam Detection
- Scenario: Classify emails as spam or not.

- Challenge: Minimizing false positives is critical, as important emails misclassified as spam could harm users.

- Metric Priority: High precision to avoid misclassifying legitimate emails as spam.

- Threshold Selection:
	+ Increase the threshold to prioritize precision.
	+ Example: Flag emails as spam only if the probability > 0.8.
	+ Trade-Off: Some spam emails might not be detected (lower recall).

4. Use Case: Autonomous Vehicles
- Scenario: Detect obstacles in the path of a self-driving car.

- Challenge: False negatives (failing to detect an obstacle) could lead to accidents, while false positives might lead to unnecessary stops.

- Metric Priority: Balance between precision and recall (F1 Score).

- Threshold Selection:
	+ Use a balanced threshold (e.g., 0.5) to ensure both metrics are optimized.
	+ Dynamic thresholds can be applied based on environmental conditions.




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

## PCA


## t-SNE
- t-distributed stochastic neighbor embedding
- visualizing high-dimensional data in a lower-dimensional space
- Objective Function: KL divergence
	+ P: The probability distribution of pairwise distances in the high-dimensional space.
	+ Q: The probability distribution of pairwise distances in the low-dimensional space.
	+ minimize $$\text{KL}(P \| Q) =\sum_{i=1}^N \sum_{j=1}^N P_{ij} \log \frac{P_{ij}}{Q_{ij}} $$
		+ $P_{ij}$: probability point $i$ and $j$ are neighbors in the high-dimensional space
		+ $Q_{ij}$: probability point $i$ and $j$ are neighbors in the low-dimensional space
	+ Distances in high dimensional space
		$$P_{ij} = \frac{e^{-\|x_i-x_j\|^2 / 2\sigma^2}}{\sum_{k\not=i} e^{-\|x_i-x_k\|^2 / 2\sigma^2}}$$
		+ it measures similarity between $x_i$ and $x_j$
		+ it is the probability that i will pick j as its neighbor under Gaussian distribution
	+ Distances in low dimensional space
		$$Q_{ij} =\frac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{k\not=i}(1+\|y_i-y_k\|^2)^{-1}}$$
		+ it is distance in heavy-tailed Student t-distribution
- stochastic optimization via gradient descent
	+ the gradient (derivation: https://stats.stackexchange.com/questions/301276/calculating-t-sne-gradient-a-mistake-in-the-original-t-sne-paper)
		$$\frac{\partial \text{KL}(P \| Q)}{\partial y_i} = 4 \sum_j (P_{ij} - Q_{ij})(1+\|y_i-y_j\|^2)^{-1} (y_i-y_j)$$
	+ update $y_i$ using gradient descent
- way to determin $\sigma$, using perplexity
	+ **Perplexity**: $2^{H(p)}$, where $H$ is the entropy
	+ the larger the perplexity, the less likely it is that an observer can guess the value
	+ adjust $\sigma_i$ for each row $P_{i\cdot}$ until reach the target perplexity
	+ use a binary search to get the desired sigma.

- Why t-distribution:
	+ heavy tail property, reducing impact of outliers, allowing some distances to be less faithfully preserved in the embedding




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


## Kernel Based Methods

### Support Vector Machine (SVM)
**Problem**
Given data $\{x_i,y_i\|_{i=1}^N$ where $y_i\in \{-1,1\}$ is the class label. Find a plane $w\cdot x + b = 0$ that separates the two classes with the largest margin.
- margin: perpendicular distance between the two closest points of the classes and the hyperplane
	+ $wx+b=1$ and $wx+b=-1$ distance $$d = \text{margin} = \frac{2}{\|w\|}$$

In mathematical expression, the problem is to optimize
$$\min_{w,b}\frac12 \|w\|^2 \text{ such that } y_i (w\cdot x_i + b) \ge 1 \text{ for any }i=1,\ldots,N.$$

**Solution by Lagrangian Dual Formulation**
- Lagrangian dual formulation $\mathcal L(w,b,\alpha) = \frac12 \|w\|^2  - \sum_{i=1}^N \alpha_i[y_i(w\cdot x_i +b) - 1] $ 
	+ critical point satisfies 
		$$\frac{\partial \mathcal L}{\partial w} =  w - \sum_{i=1}^N \alpha_i y_i x_i = 0$$
		$$\frac{\partial \mathcal L}{\partial b} = \sum_{i=1}^N \alpha_i y_i = 0$$
	+ substitute the $w$ in to $L$ gives the dual problem 
		$$\max_\alpha \sum_{i=1}^N \alpha_i - \frac12 \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i\cdot x_j)$$
		$$\text{subject to }\sum_{i=1}^N \alpha_i y_i = 0, \alpha_i\ge 0.$$
	+ solve the dual problem to find $\alpha$ and $w = \sum_{i=1}^N \alpha_i y_i x_i, b = y_k - \sum_{i=1}^N \alpha_i y_i (x_i\cdot x_k)$

- for nonlinear SVM, the plane is given by 
	$$y = w\cdot \varphi(x) + b$$
	+ the dual problem is 
		$$\max_\alpha \sum_{i=1}^N \alpha_i - \frac12 \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j \varphi(x_i)\cdot \varphi(x_j)$$
	+ the kernel $K(x_i,x_j) =\varphi(x_i)\cdot \varphi(x_j) $
	+ usually, the function $\varphi$ is difficult to find and $K$ is easy to define
	+ in the nonlinear case, $w = \sum_{i=1}^N \alpha_i y_i \varphi(x_i)$ , $b_k = y_k - \sum_{i=1}^N \alpha_i K(x_i,x_k)$ and $y=w\cdot \varphi(x) +b$ where 
	$$w\cdot \varphi(x) = \sum_{i=1}^N \alpha_i y_i \varphi(x_i) \cdot \varphi(x) =  \sum_{i=1}^N \alpha_i y_i K(x_i,x)  $$

**Two Approaches**

There are two approaches for solving the SVM, by solving the primal problem or the dual problem. Of course, we need a duality theorem to show there are equivalent. We will not touch this issue here but assume they are equivalent.

**Solution**
- If function $\varphi$ is given, we can solve the problem by 
	$$\min_{w} L := \lambda \|w\|^2 +  \frac1N \sum_{i=1}^N \max(0, 1- y_i(w\cdot \varphi(x_i)+b))$$
	+ We use a penalty parameter $\lambda$ for imposing the constraint, the optimal value for $\lambda$ is zero (it is better to put $1/\lambda$ in front of the second term to demonstrate this). 
	+ The gradients
		$$\frac{\partial L}{\partial w} = 2 \lambda w - \frac1N 1_{y_i(w\cdot \varphi(x_i)+b) \ge 1} y_i \varphi(x_i) $$
		$$\frac{\partial L}{\partial b} = - \frac1N 1_{y_i(w\cdot \varphi(x_i)+b) \ge 1} y_i$$
	+ Update rule
	$$w \gets w - \eta \frac{\partial L}{\partial w}$$
	$$b \gets b - \eta \frac{\partial L}{\partial b}$$
	+ It should be figured out that the second term in the loss function is called **Hinge Loss**, which is actually non-differentiable. What we use is a sub-gradient.

- When $\varphi$ is not known. We use the kernel 
	$$y = w\cdot \varphi(x) + b = \sum_{i=1}^N \alpha_i y_i K(x_i,x) + b$$
	and 
	$$\|w\|^2 = \|\sum_{i} \alpha_i y_i \varphi(x_i)\|^2 = \sum_{i}\sum_j \alpha_i \alpha_j y_i y_j K(x_i,x_j)$$
	+ Loss function
	$$L = \lambda  \sum_{i}\sum_j \alpha_i \alpha_j y_i y_j K(x_i,x_j) + \frac1N \sum_{i=1}^N \max(0, 1- y_i(\sum_{i=1}^N \alpha_i y_i K(x_i,x) + b)) $$
	+ One can solve this problem in the similar manner, we will not provide the code here.

- One can also solve the dual problem. 
	+ loss $$L =\sum_{i=1}^N \alpha_i - \frac12 \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i,x_j) $$
	+ gradients
	$$\frac{\partial L}{\partial \alpha_i} = 1- \sum_j \alpha_j y_iy_j K(x_i,x_j)$$
	+ gradient ascent (maximize)
	$$\alpha \gets \alpha + \eta \frac{\partial L}{\partial \alpha} $$
	+ to impose $\sum_i \alpha_i y_i = 0$, we compute $\Delta = \sum_{i} \alpha_i y_i$ and update $\alpha_i$ with 
	$$\alpha_i = \alpha_i - \frac{\Delta}{y_i}$$
	+ We also impose a maximum value for $\alpha \le C$ by clipping 
	$$\alpha_i = \max(\min(\alpha_i,0),C)$$

- Support vector machine use the **Hinge loss** of the form $\max(0, 1-t\cdot y)$.
	+ Hinge Loss is more robust to outliers
	+ Hinge loss can be more stable for large-margin classifiers



