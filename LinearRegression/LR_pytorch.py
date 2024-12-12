import torch 
import torch.nn as nn 


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias = bias)
    
    def forward(self, X):
        return self.linear(X)
    
    def optimizer(self, learning_rate=1e-3):
        return torch.optim.SGD(self.parameters(), lr = learning_rate)
    
    def train(self, X, y, learning_rate=1e-3, n_iterations = 10000):
        optimizer = self.optimizer(learning_rate)
        losses = []
        for i in range(n_iterations):
            y_pred = self.forward(X)
            loss = torch.mean((y_pred - y)**2)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
        return losses

    def predict(self, X):
        return self.linear(X)
    



if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt 

    n_samples = 100
    n_features = 10
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train= torch.tensor(y_train, dtype=torch.float32)[:, None]
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)[:, None]

    # Train the model
    model = LinearLayer(n_features, 1)
    losses = model.train(X_train, y_train)
    plt.plot(losses)
    plt.show()

    # Predict
    y_pred = model.predict(X_test)

    plt.scatter(X_test[:,0], y_test, c='blue', marker='^')
    plt.scatter(X_test[:,0], y_pred.detach(), c='red')
    plt.show()

