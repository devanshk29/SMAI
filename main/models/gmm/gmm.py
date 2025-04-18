import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_components, tol=1e-6, max_iter=100, random_state=None):
        self.n_components = n_components  
        self.tol = tol  
        self.max_iter = max_iter  
        self.random_state = random_state
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.n_samples = None
        self.n_features = None 

    def _initialize_parameters(self, X):
        np.random.seed(self.random_state)
        self.n_samples, self.n_features = X.shape
        self.means_ = X[np.random.choice(self.n_samples, self.n_components, False)]
        self.covariances_ = np.array([np.eye(self.n_features)] * self.n_components)
        self.weights_ = np.ones(self.n_components) / self.n_components

    def _m_step(self, X, responsibilities):
        """Maximization step"""
        nk = np.sum(responsibilities, axis=0)
        self.weights_ = nk / self.n_samples
        self.means_ = np.dot(responsibilities.T, X) / (nk[:, np.newaxis]+1e-10)
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = np.dot(responsibilities[:, k] * diff.T, diff) / (nk[k]+1e-10)
            self.covariances_[k] += 1e-6 * np.eye(self.n_features)
    
    def _e_step(self, X):
        """Expectation step using multivariate normal PDF"""
        log_prob = np.zeros((self.n_samples, self.n_components))
        for k in range(self.n_components):
            mvn = multivariate_normal(mean=self.means_[k], cov=self.covariances_[k])
            log_prob[:, k] = np.log(self.weights_[k]) + mvn.logpdf(X) 
        
        prob = np.exp(log_prob)

        prob[prob == 0] = 0

        responsibilities = prob / (np.sum(prob, axis=1, keepdims=True))

        responsibilities[np.isnan(responsibilities)] = 0

        return responsibilities


    def fit(self, X):
        self._initialize_parameters(X)
        log_likelihood_old = -np.inf
        for iteration in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            
            log_likelihood = np.sum(np.log(np.sum(np.exp(self._e_step(X)), axis=1) + 1e-3))
            # print(f"Iteration {iteration + 1}: Log Likelihood = {log_likelihood}")
         

    def getParams(self):
        return self.weights_, self.means_, self.covariances_
        # return {'means': self.means_, 'covariances': self.covariances_, 'weights': self.weights_}

    def getMembership(self, X):
        responsibilities = self._e_step(X)
        return responsibilities

    def getLikelihood(self, X):
        responsibilities = self._e_step(X)
        likelihood = np.sum(np.log(np.sum(responsibilities * self.weights_, axis=1) + 1e-10))
        return likelihood

    def predict(self, X):
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
        