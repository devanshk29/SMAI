import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        eigenvalues = eigenvalues[sorted_indices]
        
        self.components_ = eigenvectors[:, :self.n_components]
        
        self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def checkPCA(self, X, threshold=1e-10):
        self.fit(X)
        
        cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio_)
        
        information_retained = cumulative_variance_ratio[self.n_components - 1]
        information_loss = 1 - information_retained
        
        if information_loss > threshold:
            return False
           
        
        X_transformed = self.transform(X)
        
        if X_transformed.shape != (X.shape[0], self.n_components):
            return False
           
        
        return True