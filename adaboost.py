import numpy as np 


class AdaBoost:
    def __init__(self, weak_learner, n_estimators=50):
        self.weak_learner = weak_learner
        self.n_estimators = n_estimators
        self.betas = []
        self.learners = []
    
    def fit(self, X, y):
        n_samples, _ = X.shape
        weights = np.ones(n_samples) / n_samples # initial weight 

        for estimator_idx in range(self.n_estimators):
            learner = self.weak_learner()
            learner.fit(X, y, weights)
            y_pred = learner.predict(X)
            missclassified = (y_pred != y)
            err = np.sum(weights * missclassified) / np.sum(weights)
            if err > 0.5:
                break 
            elif err == 0:
                self.learners.append(learner)
                self.betas.append(1)
                break 

            # compute beta
            beta = 0.5 * np.log( (1-err)/err )
            self.learners.append(learner)
            self.betas.append(beta)

            # update weights
            weights *= np.exp(-beta*y*y_pred)
            weights /= np.sum(weights)
        
    def predict(self, X):
        prediction = np.zeros(X.shape[0])
        for beta, learner in zip(self.betas, self.learners):
            prediction += beta * learner.predict(X)
        return np.sign(prediction)
