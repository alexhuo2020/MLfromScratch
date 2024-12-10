import numpy as np 


class SoftmaxRegression:
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None 
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        self.losses = []

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias 
            
            y_pred = self._softmax(linear_model)
            y_true_one_hot = np.eye(n_classes)[y]

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y_true_one_hot))
            db = (1/n_samples) * np.sum(y_pred - y_true_one_hot, axis=0) 

            self.weights -= self.learning_rate * dw 
            self.bias -= self.learning_rate * db 

            self.losses.append(self._CE_loss(np.argmax(y_pred, axis=1), y_true_one_hot))
        
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias 
        y_pred = self._softmax(linear_model)
        return np.argmax(y_pred, axis=1)      

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stabilize with max subtraction, first dim batch dim
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    
    def _CE_loss(self, y_true, y_pred):
        n_samples = y_true.shape[0]
        y_true_one_hot = np.eye(y_pred.shape[1])[y_true]  # Convert to one-hot encoding
        epsilon = 1e-15  # To avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true_one_hot * np.log(y_pred)) / n_samples
        
    



if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, n_informative=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = SoftmaxRegression(learning_rate=0.01, n_iterations=200)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    import matplotlib.pyplot as plt 
    plt.plot(model.losses)
    plt.show()




