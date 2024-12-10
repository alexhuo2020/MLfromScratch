import numpy as np 

class LinearRegression:
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None 
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias 
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y) 
            self.weights -= self.learning_rate * dw 
            self.bias -= self.learning_rate * db 
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias 
    

class LinearRegression2:
    def __init__(self):
        self.weights = None 
        self.bias = None 
    
    def fit(self, X, y):
        X_a = np.c_[np.ones((X.shape[0], 1)), X]  # Add x0 = 1 to each sample
        beta_best = np.linalg.inv(X_a.T @ X_a) @ X_a.T @ y
        self.bias, self.weights = beta_best[0], beta_best[1]     
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias 
    




if __name__ == '__main__':
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])  # Perfectly linear relationship (y = 2x)

    # Train the model
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)
    print("Predicted values:", predictions)
    print("Weights:", model.weights)
    print("Bias:", model.bias)

    model = LinearRegression2()
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)
    print("Predicted values:", predictions)
    print("Weights:", model.weights)
    print("Bias:", model.bias)

