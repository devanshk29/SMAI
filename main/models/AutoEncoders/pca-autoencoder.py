
import numpy as np

class PcaAutoencoder:
    def __init__(self, n_components):
        """
        Initialize the PCA Autoencoder with the number of components (n_components) to retain.
        """
        self.n_components = n_components
        self.mean = None
        self.eigenvectors = None

    def fit(self, X):
        """
        Calculate eigenvalues and eigenvectors from the input data.
        Parameters:
            X (numpy array): The input data matrix where each row is an observation.
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Calculate the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Select the top n_components eigenvectors
        self.eigenvectors = eigenvectors[:, :self.n_components]

    def encode(self, X):
        """
        Reduce the dimensionality of the input data using the learned eigenvectors.
        Parameters:
            X (numpy array): The input data matrix to be encoded.
        Returns:
            numpy array: The reduced-dimensional representation of X.
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.eigenvectors)

    def forward(self, X):
        """
        Reconstruct the data from the reduced representation.
        Parameters:
            X (numpy array): The input data matrix to be reconstructed.
        Returns:
            numpy array: The reconstructed data, in the original dimensionality.
        """
        # Encode data to lower dimension
        X_reduced = self.encode(X)
        
        # Reconstruct data from lower dimension
        X_reconstructed = np.dot(X_reduced, self.eigenvectors.T) + self.mean
        return X_reconstructed
