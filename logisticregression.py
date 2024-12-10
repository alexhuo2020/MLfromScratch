import numpy as np 


class LogisticRegression:
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None 
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias 
            y_pred = self._sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y) 

            self.weights -= self.learning_rate * dw 
            self.bias -= self.learning_rate * db 
            self.losses.append(self._BCE_loss(y_pred, y))
        
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias 
        y_pred = self._sigmoid(linear_model)
        return np.round(y_pred)
    

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _BCE_loss(self, y_true, y_pred):
        epsilon = 1e-15  # To avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
    



if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression(learning_rate=0.01, n_iterations=2000)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    import matplotlib.pyplot as plt 
    plt.plot(model.losses)
    plt.show()




