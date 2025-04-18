import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from models.MLP.MLPreg import MLPRegressor
from models.MLP.MLP import UnifiedMLP
class AutoEncoder:
    def __init__(self, input_size, hidden_layers, latent_dim, lr=0.01, epochs=100, batch_size=32, activation="relu", optimizer="sgd"):
        """
        Initializes the autoencoder model with one MLP.
        The network reduces to `latent_dim` in the middle and reconstructs the input.
        
        Args:
        - input_size (int): Dimension of input vector (12 for your case).
        - hidden_layers (list): List of hidden layer sizes.
        - latent_dim (int): Dimension to reduce the input to (9 in your case).
        - lr (float): Learning rate.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        - activation (str): Activation function to use.
        - optimizer (str): Optimizer to use.
        """
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.optimizer = optimizer

        # Define the architecture of the autoencoder with one network
        # The hidden layers will first reduce to the latent dimension and then expand back to the input size
        self.mlp = UnifiedMLP(input_size=input_size, 
                                hidden_layers=hidden_layers + [latent_dim] + list(reversed(hidden_layers)), 
                                output_size=input_size, 
                                lr=lr, epochs=epochs, batch_size=batch_size, 
                                activation=activation, optimizer=optimizer, task="regression")

    def fit(self, X):
        """
        Trains the autoencoder to reconstruct the input data.
        
        Args:
        - X (np.ndarray): The input dataset with shape (num_samples, input_size).
        """
        # Train the MLP to act as an autoencoder (encoder + decoder combined)
        self.mlp.fit(X, X)  # Target is also X, because we're reconstructing the input

    def get_latent(self, X):
        """
        Returns the latent representation of the dataset (extracted from the bottleneck layer).
        
        Args:
        - X (np.ndarray): Input data to encode.
        
        Returns:
        - np.ndarray: The reduced dataset (latent space representation).
        """
        # Forward pass up to the latent dimension layer
        current_output = X
        for i, weight in enumerate(self.mlp.weights):
            current_output = np.dot(current_output, weight) + self.mlp.biases[i]
            current_output = self.mlp._activate(current_output)
            # Check if we reached the latent dimension layer
            if weight.shape[1] == self.latent_dim:
                break

        return current_output
