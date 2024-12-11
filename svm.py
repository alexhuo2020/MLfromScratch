import numpy as np 

class SVMslack:
    def __init__(self, slack_fn = lambda x: x, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.phi = slack_fn
    
    def fit(self, X, y):
        n_samples, n_features = X.shape 
        self.weights = np.zeros(n_features)
        self.bias = 0 
        y_ = np.where(y <=0, -1, 1) # make sure label is -1, 1
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(self.phi(x_i), self.weights) + self.bias) >=1
                if condition:
                    dw = 2 * self.lambda_param * self.weights 
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.weights - np.dot(self.phi(x_i), y_[idx]) / n_samples
                    db = - y_[idx] / n_samples
                self.weights -= self.learning_rate * dw 
                self.bias -= self.learning_rate * db 
    
    def predict(self, X):
        approax = np.dot(self.phi(X), self.weights) + self.bias 
        return np.sign(approax)
    

class SVM:
    """use kernel"""
    def __init__(self, kernel = 'rbf', gamma=0.1, C = 10, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  
        self.n_iters = n_iters
        self.alpha = None
        self.bias = None
        self.kernel = kernel
        self.gamma = gamma
        self.C = C

    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_kernel_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])
        return K

    def fit(self, X, y):
        n_samples, n_features = X.shape 
        self.alpha = np.zeros(n_samples)
        K = self._compute_kernel_matrix(X)

        y_ = np.where(y <=0, -1, 1) # make sure label is -1, 1
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                dalpha = 1 - np.sum(self.alpha * y_ * K[:, idx]) * y_[idx]
                self.alpha += self.learning_rate * dalpha 
                self.alpha = np.clip(self.alpha, 0, self.C)
                self.alpha -= np.sum(self.alpha * y_) / y_ # make sure the constraint \sum_i \alpha_i y_i = 0

        self.support_vector_indices = np.where(self.alpha > 1e-4)[0]
        self.support_vector = X[self.support_vector_indices]
        self.support_vector_labels = y[self.support_vector_indices]
        self.bias = np.mean([
            y[i] - np.sum(self.alpha * y * K[:, i])
            for i in self.support_vector_indices
        ])

    
    def predict(self, X):
        y_pred = []
        for x in X:
            prediction = np.sum(
                self.alpha[self.support_vector_indices]  * self.support_vector_labels *
                [self._kernel_function(sv, x) for sv in self.support_vector]
            ) + self.bias
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)
    
if __name__ == "__main__":
    # Toy dataset
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [5, 5],
        [1, 0],
        [0, 1],
        [6, 6]
    ])
    y = np.array([-1, -1, -1, 1, 1, 1, 1])

    # Train SVM
    # svm = SVMslack(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
    svm = SVM(kernel='rbf',gamma=1,learning_rate=0.01, lambda_param=0.01, n_iters=1000)
    svm.fit(X, y)
    predictions = svm.predict(X)

    print("Predictions:", predictions)
    # print("Weights:", svm.weights)
    # print("Bias:", svm.bias)
    # from sklearn.svm import LinearSVC
    # svm = LinearSVC()

    # svm.fit(X, y)
    # predictions = svm.predict(X)

    # print("Predictions:", predictions)
    # print("Weights:", svm.coef_)
    # print("Bias:", svm.intercept_)



    


