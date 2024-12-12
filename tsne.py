import numpy as np 

class TSNE:
    def __init__(self, n_components, perplexity, learning_rate, n_iter):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate 
        self.n_iter = n_iter 
        self.sigma = None

    def fit(self, X):
        n_samples, n_features = X.shape
        distances =  np.linalg.norm(X[:, np.newaxis] - X,axis=-1)


        self.sigma = self._compute_sigma(distances, self.perplexity)
        P = np.zeros_like(distances)

        ## Get P prob
        for i in range(n_samples):
            P[i] = np.exp(-distances[i]**2 / (2 * self.sigma[i]**2))
            P[i, i] = 0  # Set self-probability to 0
            P[i] /= np.sum(P[i])

        # Symmetrize P
        P = (P + P.T) / (2 * n_samples)
        P = np.maximum(P, 1e-12)  # Prevent numerical instability

        ## initialize low-dimensional embdding
        Y = np.random.randn(n_samples, self.n_components) * 0.01

        for iter in range(self.n_iter):
            low_distances = np.linalg.norm(Y[:, np.newaxis] - Y,axis=-1)
            Q = 1 / (1 + low_distances**2)
            np.fill_diagonal(Q, 0)  # Set Q_ii to 0
            Q /= np.sum(Q)#, axis=1, keepdims=True)
            Q = np.maximum(Q, 1e-12)  # Prevent numerical instability

            # Compute gradients
            PQ_diff = P - Q
            dY = np.zeros_like(Y)
            for i in range(n_samples):
                dY[i] = np.sum(
                    (PQ_diff[i, :, np.newaxis] * (Y[i] - Y)) / (1 + low_distances[i][:, np.newaxis]**2),
                    axis=0
                )
            dY *= 4  # Scaling factor from the TSNE equation

            # Update Y
            Y -= self.learning_rate * dY
        return Y 
    
    def _compute_sigma(self, distances, perplexity, tol=1e-5, max_iter=50):
        n_samples = distances.shape[0]
        sigmas = np.ones(n_samples)
        target_entropy = np.log2(perplexity)

        for i in range(n_samples):
            sigma_min, sigma_max = 1e-5, 1e5
            sigma = 1.0

            for _ in range(max_iter):
                # Compute conditional probabilities
                P = np.exp(-distances[i]**2 / (2 * sigma**2))
                P[i] = 0  # Set self-probability to 0
                P /= np.sum(P)
                entropy = -np.sum(P * np.log2(P + 1e-10))  # Compute entropy

                # Adjust sigma using binary search
                if np.abs(entropy - target_entropy) < tol:
                    break
                if entropy > target_entropy:
                    sigma_max = sigma
                else:
                    sigma_min = sigma
                sigma = (sigma_min + sigma_max) / 2.0

            sigmas[i] = sigma

        return sigmas

if __name__ == "__main__":
    # Generate sample high-dimensional data
    n_samples, n_features = 100, 50  # 100 samples, 50 features
    X = np.random.randn(n_samples, n_features)

    # Perform t-SNE with a specific perplexity
    perplexity = 30
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=80, n_iter=1000)
    low_dim_embeddings = tsne.fit(X)
    import matplotlib.pyplot as plt 
    # Visualize the result
    from sklearn.manifold import TSNE as TSNE_sklearn
    Y = TSNE_sklearn().fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], c='red', marker='^', alpha=0.7)

    plt.scatter(low_dim_embeddings[:, 0], low_dim_embeddings[:, 1], c='blue', alpha=0.7)
   



    plt.show()
    
