import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from sklearn.datasets import make_blobs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.gmm.gmm import GMM

class KDE:
    def __init__(self, kernel='gaussian', bandwidth=1.0):
        self.kernel = kernel
        self.bandwidth = bandwidth

    def _kernel_function(self, u):
        """Kernel function based on the chosen kernel type."""
        if self.kernel == 'gaussian':
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
        elif self.kernel == 'box':
            return 0.5 * (np.abs(u) <= 1)
        elif self.kernel == 'triangular':
            return (1 - np.abs(u)) * (np.abs(u) <= 1)
        else:
            raise ValueError("Unsupported kernel. Choose 'gaussian', 'box', or 'triangular'.")

    def fit(self, X):
        """Fit the data for KDE."""
        self.X = X
        self.n, self.d = X.shape

    def predict(self, x):
        """Predict density at a given point x."""
        density = 0
        for xi in self.X:
            u = (x - xi) / self.bandwidth
            density += self._kernel_function(np.linalg.norm(u))
        return density / (self.n * self.bandwidth**self.d)

    def visualize(self, grid_size=100):
        """Visualize the KDE density for 2D data."""
        if self.X.shape[1] != 2:
            raise ValueError("Visualization is only supported for 2D data.")
        
        # Create a grid of points
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        x_grid, y_grid = np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]

        # Compute density for each point in the grid
        density = np.array([self.predict(point) for point in grid_points])
        density = density.reshape(grid_size, grid_size)

        # Plot density as a contour plot
        plt.contourf(X_grid, Y_grid, density, cmap='viridis', levels=30)
        plt.colorbar(label="Density")
        plt.scatter(self.X[:, 0], self.X[:, 1], c='red', s=5, label="Data Points")
        plt.title(f"KDE Density Plot using {self.kernel.capitalize()} Kernel")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.show()



# Usage
np.random.seed(0)
# Generate some 2D synthetic data
X = np.vstack([
    np.random.normal([2, 2], 0.5, size=(100, 2)),
    np.random.normal([-2, -2], 0.5, size=(100, 2))
])

# Instantiate KDE with Gaussian kernel
kde = KDE(kernel='gaussian', bandwidth=0.5)
kde.fit(X)

# Predict density at a given point
x_test = np.array([0, 0])
print(f"Density at {x_test}: {kde.predict(x_test)}")

# Visualize the KDE density for the data
kde.visualize()



import numpy as np
import matplotlib.pyplot as plt

# Function to generate points within a circle
def generate_circle_data(num_points, radius, center=(0, 0), noise=0.1):
    angles = 2 * np.pi * np.random.rand(num_points)
    radii = radius * np.sqrt(np.random.rand(num_points))
    x = center[0] + radii * np.cos(angles) + noise * np.random.randn(num_points)
    y = center[1] + radii * np.sin(angles) + noise * np.random.randn(num_points)
    return np.column_stack((x, y))

# Generate data for the larger, diffused circle
large_circle = generate_circle_data(num_points=3000, radius=3, center=(0, 0), noise=0.2)

# Generate data for the smaller, dense circle
small_circle = generate_circle_data(num_points=500, radius=0.5, center=(1, 1), noise=0.05)

# Combine the data
data = np.vstack((large_circle, small_circle))

# Plot the data
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], s=1, color="black")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title("Synthetic dataset with extreme overlap between two regions of density")
plt.grid(True)
plt.show()

class KDE:
    def __init__(self, kernel='gaussian', bandwidth=1.0):
        self.kernel = kernel
        self.bandwidth = bandwidth

    def _kernel_function(self, u):
        if self.kernel == 'gaussian':
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
        elif self.kernel == 'box':
            return 0.5 * (np.abs(u) <= 1)
        elif self.kernel == 'triangular':
            return (1 - np.abs(u)) * (np.abs(u) <= 1)
        else:
            raise ValueError("Unsupported kernel. Choose 'gaussian', 'box', or 'triangular'.")

    def fit(self, X):
        self.X = X
        self.n, self.d = X.shape

    def predict(self, x):
        density = 0
        for xi in self.X:
            u = (x - xi) / self.bandwidth
            density += self._kernel_function(np.linalg.norm(u))
        return density / (self.n * self.bandwidth**self.d)

    def visualize(self, grid_size=100):
        if self.X.shape[1] != 2:
            raise ValueError("Visualization is only supported for 2D data.")
        
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        x_grid, y_grid = np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]

        density = np.array([self.predict(point) for point in grid_points])
        density = density.reshape(grid_size, grid_size)

        plt.contourf(X_grid, Y_grid, density, cmap='viridis', levels=30)
        plt.colorbar(label="Density")
        plt.scatter(self.X[:, 0], self.X[:, 1], c='red', s=5, label="Data Points")
        plt.title(f"KDE Density Plot using {self.kernel.capitalize()} Kernel")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Function to generate points within a circle
def generate_circle_data(num_points, radius, center=(0, 0), noise=0.1):
    angles = 2 * np.pi * np.random.rand(num_points)
    radii = radius * np.sqrt(np.random.rand(num_points))
    x = center[0] + radii * np.cos(angles) + noise * np.random.randn(num_points)
    y = center[1] + radii * np.sin(angles) + noise * np.random.randn(num_points)
    return np.column_stack((x, y))

# Generate data for the larger, diffused circle
large_circle = generate_circle_data(num_points=3000, radius=3, center=(0, 0), noise=0.2)

# Generate data for the smaller, dense circle
small_circle = generate_circle_data(num_points=500, radius=0.5, center=(1, 1), noise=0.05)

# Combine the data
data = np.vstack((large_circle, small_circle))



kde = KDE(kernel='gaussian', bandwidth=0.5)
kde.fit(data)
kde.visualize()



from sklearn.mixture import GaussianMixture

# Fit a GMM with 2 components
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
gmm.fit(data)

# Predict the density for a grid of points
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = gmm.score_samples(XX)  # GMM returns log-likelihood, invert it for density
Z = Z.reshape(X.shape)

# Plot GMM density
plt.contourf(X, Y, Z, levels=30, cmap='viridis')
# plt.scatter(data[:, 0], data[:, 1], c='red', s=5)
plt.colorbar(label="Density")
plt.title("GMM Density Plot with 2 Components")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


# Plot GMM density
plt.contourf(X, Y, Z, levels=30, cmap='viridis')
# plt.scatter(data[:, 0], data[:, 1], c='red', s=5)
plt.colorbar(label="Density")
plt.title("GMM Density Plot with 2 Components")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()



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
        

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Function to generate points within a circle
def generate_circle_data(num_points, radius, center=(0, 0), noise=0.1):
    angles = 2 * np.pi * np.random.rand(num_points)
    radii = radius * np.sqrt(np.random.rand(num_points))
    x = center[0] + radii * np.cos(angles) + noise * np.random.randn(num_points)
    y = center[1] + radii * np.sin(angles) + noise * np.random.randn(num_points)
    return np.column_stack((x, y))

# Generate data for the larger, diffused circle
large_circle = generate_circle_data(num_points=3000, radius=3, center=(0, 0), noise=0.2)

# Generate data for the smaller, dense circle
small_circle = generate_circle_data(num_points=500, radius=0.5, center=(1, 1), noise=0.05)

# Combine the data
data = np.vstack((large_circle, small_circle))

# Use the custom GMM class
gmm = GMM(n_components=2, max_iter=100, tol=1e-6, random_state=42)
gmm.fit(data)

# Get GMM parameters
weights, means, covariances = gmm.getParams()

# Create a grid for density visualization
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T

# Compute the density for each point in the grid
Z = np.zeros(XX.shape[0])
for k in range(gmm.n_components):
    mvn = multivariate_normal(mean=means[k], cov=covariances[k])
    Z += weights[k] * mvn.pdf(XX)

# Reshape Z for plotting
Z = Z.reshape(X.shape)

# Plot GMM density
plt.contourf(X, Y, Z, levels=30, cmap='viridis')
plt.scatter(data[:, 0], data[:, 1], c='red', s=5)
plt.colorbar(label="Density")
plt.title("GMM Density Plot with 2 Components")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

plt.contourf(X, Y, Z, levels=30, cmap='viridis')
# plt.scatter(data[:, 0], data[:, 1], c='red', s=5)
plt.colorbar(label="Density")
plt.title("GMM Density Plot with 2 Components")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Function to generate points within a circle
def generate_circle_data(num_points, radius, center=(0, 0), noise=0.1):
    angles = 2 * np.pi * np.random.rand(num_points)
    radii = radius * np.sqrt(np.random.rand(num_points))
    x = center[0] + radii * np.cos(angles) + noise * np.random.randn(num_points)
    y = center[1] + radii * np.sin(angles) + noise * np.random.randn(num_points)
    return np.column_stack((x, y))

# Generate data for the larger, diffused circle
large_circle = generate_circle_data(num_points=3000, radius=3, center=(0, 0), noise=0.2)

# Generate data for the smaller, dense circle
small_circle = generate_circle_data(num_points=500, radius=0.5, center=(1, 1), noise=0.05)

# Combine the data
data = np.vstack((large_circle, small_circle))

# Helper function to fit GMM and plot density
def fit_and_visualize_gmm(data, n_components, grid_range=(-4, 4), grid_size=100):
    gmm = GMM(n_components=n_components, max_iter=100, tol=1e-6, random_state=42)
    gmm.fit(data)

    # Get GMM parameters
    weights, means, covariances = gmm.getParams()

    # Create a grid for density visualization
    x = np.linspace(grid_range[0], grid_range[1], grid_size)
    y = np.linspace(grid_range[0], grid_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    # Compute the density for each point in the grid
    Z = np.zeros(XX.shape[0])
    for k in range(gmm.n_components):
        mvn = multivariate_normal(mean=means[k], cov=covariances[k])
        Z += weights[k] * mvn.pdf(XX)

    # Reshape Z for plotting
    Z = Z.reshape(X.shape)

    # Plot GMM density
    plt.contourf(X, Y, Z, levels=30, cmap='viridis')
    # plt.scatter(data[:, 0], data[:, 1], c='red', s=5, label="Data Points")
    plt.colorbar(label="Density")
    plt.title(f"GMM Density Plot with {n_components} Components")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()

# Visualize with 4 components
fit_and_visualize_gmm(data, n_components=4)

# Visualize with 8 components
fit_and_visualize_gmm(data, n_components=8)


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Function to generate points within a circle
def generate_circle_data(num_points, radius, center=(0, 0), noise=0.1):
    angles = 2 * np.pi * np.random.rand(num_points)
    radii = radius * np.sqrt(np.random.rand(num_points))
    x = center[0] + radii * np.cos(angles) + noise * np.random.randn(num_points)
    y = center[1] + radii * np.sin(angles) + noise * np.random.randn(num_points)
    return np.column_stack((x, y))

# Generate data for the larger, diffused circle
large_circle = generate_circle_data(num_points=3000, radius=3, center=(0, 0), noise=0.2)

# Generate data for the smaller, dense circle
small_circle = generate_circle_data(num_points=500, radius=0.5, center=(1, 1), noise=0.05)

# Combine the data
data = np.vstack((large_circle, small_circle))

# Helper function to fit GMM and plot density
def fit_and_visualize_gmm(data, n_components, grid_range=(-4, 4), grid_size=100):
    gmm = GMM(n_components=n_components, max_iter=100, tol=1e-6, random_state=42)
    gmm.fit(data)

    # Get GMM parameters
    weights, means, covariances = gmm.getParams()

    # Create a grid for density visualization
    x = np.linspace(grid_range[0], grid_range[1], grid_size)
    y = np.linspace(grid_range[0], grid_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    # Compute the density for each point in the grid
    Z = np.zeros(XX.shape[0])
    for k in range(gmm.n_components):
        mvn = multivariate_normal(mean=means[k], cov=covariances[k])
        Z += weights[k] * mvn.pdf(XX)

    # Reshape Z for plotting
    Z = Z.reshape(X.shape)

    # Plot GMM density
    plt.contourf(X, Y, Z, levels=30, cmap='viridis')
    # plt.scatter(data[:, 0], data[:, 1], c='red', s=5, label="Data Points")
    plt.colorbar(label="Density")
    plt.title(f"GMM Density Plot with {n_components} Components")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()

# KDE visualization using the provided class
def visualize_kde(data, kernel='gaussian', bandwidth=0.5, grid_size=100):
    kde = KDE(kernel=kernel, bandwidth=bandwidth)
    kde.fit(data)
    kde.visualize(grid_size)

# Visualize with GMM (4 components)
fit_and_visualize_gmm(data, n_components=4)

# Visualize with GMM (8 components)
fit_and_visualize_gmm(data, n_components=8)

# Visualize with KDE
visualize_kde(data, kernel='gaussian', bandwidth=0.5)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Function to generate points within a circle
def generate_circle_data(num_points, radius, center=(0, 0), noise=0.1):
    angles = 2 * np.pi * np.random.rand(num_points)
    radii = radius * np.sqrt(np.random.rand(num_points))
    x = center[0] + radii * np.cos(angles) + noise * np.random.randn(num_points)
    y = center[1] + radii * np.sin(angles) + noise * np.random.randn(num_points)
    return np.column_stack((x, y))

# Generate data for the larger, diffused circle
large_circle = generate_circle_data(num_points=3000, radius=3, center=(0, 0), noise=0.2)

# Generate data for the smaller, dense circle
small_circle = generate_circle_data(num_points=500, radius=0.5, center=(1, 1), noise=0.05)

# Combine the data
data = np.vstack((large_circle, small_circle))

# Helper function to fit GMM and plot density
def fit_and_visualize_gmm(data, n_components, grid_range=(-4, 4), grid_size=100):
    gmm = GMM(n_components=n_components, max_iter=100, tol=1e-6, random_state=42)
    gmm.fit(data)

    # Get GMM parameters
    weights, means, covariances = gmm.getParams()

    # Create a grid for density visualization
    x = np.linspace(grid_range[0], grid_range[1], grid_size)
    y = np.linspace(grid_range[0], grid_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    # Compute the density for each point in the grid
    Z = np.zeros(XX.shape[0])
    for k in range(gmm.n_components):
        mvn = multivariate_normal(mean=means[k], cov=covariances[k])
        Z += weights[k] * mvn.pdf(XX)

    # Reshape Z for plotting
    Z = Z.reshape(X.shape)

    # Plot GMM density
    plt.contourf(X, Y, Z, levels=30, cmap='viridis')
    # plt.scatter(data[:, 0], data[:, 1], c='red', s=5, label="Data Points")
    plt.colorbar(label="Density")
    plt.title(f"GMM Density Plot with {n_components} Components")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()

# KDE visualization using the provided class
def visualize_kde(data, kernel='gaussian', bandwidth=0.5, grid_size=100):
    kde = KDE(kernel=kernel, bandwidth=bandwidth)
    kde.fit(data)
    kde.visualize(grid_size)

# Visualize with GMM (4 components)
fit_and_visualize_gmm(data, n_components=4)

# Visualize with GMM (8 components)
fit_and_visualize_gmm(data, n_components=8)

# Visualize with KDE
# visualize_kde(data, kernel='gaussian', bandwidth=0.5)



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Function to generate points within a circle
def generate_circle_data(num_points, radius, center=(0, 0), noise=0.1):
    angles = 2 * np.pi * np.random.rand(num_points)
    radii = radius * np.sqrt(np.random.rand(num_points))
    x = center[0] + radii * np.cos(angles) + noise * np.random.randn(num_points)
    y = center[1] + radii * np.sin(angles) + noise * np.random.randn(num_points)
    return np.column_stack((x, y))

# Generate data for the larger, diffused circle
large_circle = generate_circle_data(num_points=3000, radius=3, center=(0, 0), noise=0.2)

# Generate data for the smaller, dense circle
small_circle = generate_circle_data(num_points=500, radius=0.5, center=(1, 1), noise=0.05)

# Combine the data
data = np.vstack((large_circle, small_circle))

# Helper function to fit GMM and plot density
def fit_and_visualize_gmm(data, n_components, grid_range=(-4, 4), grid_size=100):
    gmm = GMM(n_components=n_components, max_iter=100, tol=1e-6, random_state=42)
    gmm.fit(data)

    # Get GMM parameters
    weights, means, covariances = gmm.getParams()

    # Create a grid for density visualization
    x = np.linspace(grid_range[0], grid_range[1], grid_size)
    y = np.linspace(grid_range[0], grid_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    # Compute the density for each point in the grid
    Z = np.zeros(XX.shape[0])
    for k in range(gmm.n_components):
        mvn = multivariate_normal(mean=means[k], cov=covariances[k])
        Z += weights[k] * mvn.pdf(XX)

    # Reshape Z for plotting
    Z = Z.reshape(X.shape)

    # Plot GMM density
    plt.contourf(X, Y, Z, levels=30, cmap='viridis')
    plt.colorbar(label="Density")
    # plt.scatter(data[:, 0], data[:, 1], c='red', s=5, label="Data Points")
    plt.scatter(means[:, 0], means[:, 1], c='white', edgecolor='black', s=100, marker='o', label="Cluster Centers")
    plt.title(f"GMM Density Plot with {n_components} Components")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()

# Visualize with 4 components
fit_and_visualize_gmm(data, n_components=4)

# Visualize with 8 components
fit_and_visualize_gmm(data, n_components=8)



######################################################################################################################




import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the dataset
data_path = 'archive/recordings/'

# Function to load and extract MFCCs
def extract_mfcc(file_path, n_mfcc=13):
    # Load audio file
    y, sr = librosa.load(file_path)
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs

# Function to visualize MFCC as a heatmap
# Function to visualize MFCC as a heatmap
def plot_mfcc(mfcc, title="MFCC"):
    plt.figure(figsize=(5, 2))
    sns.heatmap(mfcc, cmap='viridis', xticklabels=False, yticklabels=False, cbar_kws={'format': '%+2.0f dB'})
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.show()


# Loop over a few audio files to extract and plot MFCCs
for filename in os.listdir(data_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(data_path, filename)
        # Extract MFCC features
        mfcc = extract_mfcc(file_path)
        # Plot the MFCC heatmap
        plot_mfcc(mfcc, title=f"MFCC of {filename}")
        # Break after the first few files (optional)
        break


import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the dataset
data_path = 'archive/recordings/'

# Function to load and extract MFCCs
def extract_mfcc(file_path, n_mfcc=13):
    # Load audio file
    y, sr = librosa.load(file_path)
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs

# Function to visualize MFCC as a heatmap
# Function to visualize MFCC as a heatmap and save the plot
def plot_mfcc(mfcc, title="MFCC", save_path=None):
    plt.figure(figsize=(4, 2))
    sns.heatmap(mfcc, cmap='viridis', xticklabels=False, yticklabels=False, cbar_kws={'format': '%+2.0f dB'})
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot as {save_path}")
    else:
        plt.show()
    plt.close()


# Dictionary to store one file for each digit (0-9)
digit_files = {}

# Identify one file per digit
for filename in os.listdir(data_path):
    if filename.endswith(".wav"):
        digit = filename[0]  # Assuming filenames start with the digit label
        if digit.isdigit() and digit not in digit_files:
            digit_files[digit] = os.path.join(data_path, filename)

# Directory to save the plots
output_dir = "./figures/"
os.makedirs(output_dir, exist_ok=True)

# Plot and save MFCC for each digit
for digit, file_path in sorted(digit_files.items()):
    mfcc = extract_mfcc(file_path)
    save_path = os.path.join(output_dir, f"digit_{digit}_mfcc.png")
    plot_mfcc(mfcc, title=f"MFCC of digit {digit}", save_path=save_path)


import os
import numpy as np
import librosa
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import joblib

# Path to the dataset
data_path = 'archive/recordings/'

# Define HMM model parameters
n_components = 5   # Number of states in the HMM
n_mfcc = 13        # Number of MFCC features

# Function to extract MFCC features
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T  # Transpose to have time on the first axis

# Load dataset and organize MFCC features by digit
digit_mfccs = {str(digit): [] for digit in range(10)}
for filename in os.listdir(data_path):
    if filename.endswith(".wav"):
        digit = filename[0]  # Extract digit label from filename (e.g., "0" from "0_george_0.wav")
        file_path = os.path.join(data_path, filename)
        mfcc = extract_mfcc(file_path)
        digit_mfccs[digit].append(mfcc)

# Split dataset into training and testing sets for each digit
train_data = {}
test_data = {}
for digit, mfcc_list in digit_mfccs.items():
    train, test = train_test_split(mfcc_list, test_size=0.2, random_state=42)
    train_data[digit] = train
    test_data[digit] = test



# Train an HMM model for each digit
models = {}
for digit, mfcc_list in train_data.items():
    # Stack all MFCC arrays for the current digit
    X = np.concatenate(mfcc_list)
    lengths = [len(mfcc) for mfcc in mfcc_list]  # Lengths of individual sequences

    # Define and train the HMM model
    model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=100, random_state=42)
    model.fit(X, lengths)
    models[digit] = model

# Save models for later use (optional)
for digit, model in models.items():
    joblib.dump(model, f"hmm_digit_{digit}.pkl")



# Function to predict digit based on highest model likelihood
def predict_digit(mfcc, models):
    scores = {digit: model.score(mfcc) for digit, model in models.items()}
    predicted_digit = max(scores, key=scores.get)  # Select digit with highest score
    return predicted_digit

# Evaluate accuracy on the test set
correct = 0
total = 0
for digit, mfcc_list in test_data.items():
    for mfcc in mfcc_list:
        prediction = predict_digit(mfcc, models)
        if prediction == digit:
            correct += 1
        total += 1

accuracy = correct / total
print(f"Accuracy on test set: {accuracy * 100:.2f}%")


# Function to predict digit and print probabilities for each model
def predict_digit(mfcc, models):
    scores = {}
    for digit, model in models.items():
        try:
            score = model.score(mfcc)
        except:
            score = float('-inf')  # Handle cases where the model cannot score the input
        scores[digit] = score
        print(f"Digit {digit}: Log Probability = {score:.2f}")  # Print score for each digit model

    predicted_digit = max(scores, key=scores.get)
    print(f"Predicted Digit: {predicted_digit} with Log Probability {scores[predicted_digit]:.2f}")
    print("-" * 50)
    return predicted_digit

# Evaluate accuracy on the test set and print probabilities
correct = 0
total = 0
for digit, mfcc_list in test_data.items():
    for mfcc in mfcc_list:
        print(f"Actual Digit: {digit}")
        prediction = predict_digit(mfcc, models)
        if prediction == digit:
            correct += 1
        total += 1

accuracy = correct / total
print(f"Accuracy on test set: {accuracy * 100:.2f}%")


import os
import numpy as np
import librosa
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import joblib

# Function to predict digit based on highest model likelihood
def predict_digit(mfcc, models):
    scores = {digit: model.score(mfcc) for digit, model in models.items()}
    predicted_digit = max(scores, key=scores.get)  # Select digit with highest score
    return predicted_digit

# Step 1: Evaluate performance on provided test set
correct = 0
total = 0
for digit, mfcc_list in test_data.items():
    for mfcc in mfcc_list:
        prediction = predict_digit(mfcc, models)
        if prediction == digit:
            correct += 1
        total += 1

test_accuracy = correct / total
print(f"Accuracy on provided test set: {test_accuracy * 100:.2f}%")

# Step 2: Record and add personal recordings
# Ensure that personal recordings are stored in a separate folder, e.g., "my_recordings/"
my_recordings_path = './archive/my_recordings/'  # Path to folder with personal recordings

# Extract MFCC features from personal recordings
my_test_data = []
actual_digits = []
for filename in os.listdir(my_recordings_path):
    if filename.endswith(".wav"):
        digit = filename[0]  # Extract digit label from filename (e.g., "0" from "0_yourname.wav")
        actual_digits.append(digit)
        file_path = os.path.join(my_recordings_path, filename)
        mfcc = extract_mfcc(file_path)
        my_test_data.append((digit, mfcc))

# Step 3: Evaluate model on personal recordings
correct = 0
for actual_digit, mfcc in my_test_data:
    prediction = predict_digit(mfcc, models)
    if prediction == actual_digit:
        correct += 1

personal_accuracy = correct / len(my_test_data)
print(f"Accuracy on personal recordings: {personal_accuracy * 100:.2f}%")

# Step 4: Compare and analyze
print("\nComparison of model performance:")
print(f"Provided Test Set Accuracy: {test_accuracy * 100:.2f}%")
print(f"Personal Recordings Accuracy: {personal_accuracy * 100:.2f}%")



######################################################################################################################



import numpy as np
import pandas as pd

# Constants
num_sequences = 100000
min_length = 1
max_length = 16
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Generate dataset
def generate_dataset(num_sequences, min_length, max_length):
    sequences = []
    labels = []
    for _ in range(num_sequences):
        length = np.random.randint(min_length, max_length + 1)
        sequence = np.random.choice([0, 1], size=length).tolist()
        count_of_ones = sum(sequence)
        sequences.append(sequence)
        labels.append(count_of_ones)
    return sequences, labels

sequences, labels = generate_dataset(num_sequences, min_length, max_length)

# Convert to DataFrame for convenience
data = pd.DataFrame({'sequence': sequences, 'count': labels})

# Split the dataset
train_size = int(train_ratio * num_sequences)
val_size = int(val_ratio * num_sequences)

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Display some examples
print("Examples from the dataset:")
print(data.head())

# Save the splits
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Dataset Preparation
from torch.nn.utils.rnn import pad_sequence

class BitSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    # Pad sequences to the same length
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded_sequences, labels


# Model Definition
class BitCounterRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, rnn_type='RNN', dropout=0.0):
        super(BitCounterRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Choose the RNN type (RNN, LSTM, GRU)
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("Invalid RNN type. Choose 'RNN', 'LSTM', or 'GRU'.")

        self.fc = nn.Linear(hidden_size, 1)  # Output layer

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        if isinstance(self.rnn, nn.LSTM):  # For LSTM, initialize both h0 and c0
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x.unsqueeze(-1), (h0, c0))
        else:
            out, _ = self.rnn(x.unsqueeze(-1), h0)

        # Only the last time-step output is used
        out = self.fc(out[:, -1, :])
        return out

# Training and Evaluation
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    criterion = nn.MSELoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Optimizer

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

# Dataset Splits
train_sequences = train_data['sequence'].tolist()
train_labels = train_data['count'].tolist()
val_sequences = val_data['sequence'].tolist()
val_labels = val_data['count'].tolist()

train_dataset = BitSequenceDataset(train_sequences, train_labels)
val_dataset = BitSequenceDataset(val_sequences, val_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)



# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 1  # Binary input
hidden_size = 32  # Experiment with this
num_layers = 2  # Experiment with this
dropout = 0.2  # Experiment with this
rnn_type = 'LSTM'  # Choose 'RNN', 'LSTM', or 'GRU'

model = BitCounterRNN(input_size, hidden_size, num_layers, rnn_type, dropout).to(device)

# Training the Model
train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001)


import torch.nn.functional as F

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    criterion = nn.MSELoss()  # Using MSE loss for optimization
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            # Forward pass
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        mae = 0.0  # Initialize MAE metric
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences).squeeze()

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate MAE
                mae += F.l1_loss(outputs, labels, reduction='sum').item()

        # Average metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        mae /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, MAE: {mae:.4f}")

def random_baseline(val_loader):
    total_mae = 0.0
    for sequences, labels in val_loader:
        sequences, labels = sequences.to(device), labels.to(device)

        # Generate random counts within the range of sequence lengths
        random_preds = torch.randint(0, sequences.size(1) + 1, (sequences.size(0),)).float().to(device)

        # Calculate MAE for the random predictions
        mae = F.l1_loss(random_preds, labels, reduction='sum').item()
        total_mae += mae

    total_mae /= len(val_loader.dataset)
    print(f"Random Baseline MAE: {total_mae:.4f}")
    return total_mae

# Initialize and train the model
model = BitCounterRNN(input_size, hidden_size, num_layers, rnn_type, dropout).to(device)
train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001)

# Compare to random baseline
random_baseline(val_loader)


def generate_out_of_distribution_data(sequence_length, num_samples=1000):
    """Generate out-of-distribution data for a specific sequence length."""
    sequences = [torch.randint(0, 2, (sequence_length,)) for _ in range(num_samples)]
    labels = [seq.sum().item() for seq in sequences]
    return sequences, labels

# Generate sequences for out-of-distribution data


def evaluate_generalization(model, sequence_length, num_samples=1000):
    """Evaluate model generalization for a given sequence length."""
    sequences, labels = generate_out_of_distribution_data(sequence_length, num_samples)
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    labels = torch.tensor(labels, dtype=torch.float32)

    # Pad sequences and create DataLoader
    dataset = BitSequenceDataset(sequences, labels)
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)

    model.eval()
    total_mae = 0.0

    with torch.no_grad():
        for batch_sequences, batch_labels in loader:
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)

            # Get model predictions
            predictions = model(batch_sequences).squeeze()

            # Calculate MAE
            total_mae += F.l1_loss(predictions, batch_labels, reduction='sum').item()

    total_mae /= len(dataset)
    return total_mae

import matplotlib.pyplot as plt

# Evaluate and collect MAE for different sequence lengths
sequence_lengths = range(1, 33)
mae_scores = []

for length in sequence_lengths:
    mae = evaluate_generalization(model, sequence_length=length, num_samples=1000)
    mae_scores.append(mae)
    print(f"Sequence Length: {length}, MAE: {mae:.4f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sequence_lengths, mae_scores, marker='o', label="MAE")
plt.title("Model Generalization Across Sequence Lengths")
plt.xlabel("Sequence Length")
plt.ylabel("Mean Absolute Error (MAE)")
plt.grid(True)
plt.legend()
plt.savefig("generalization_plot.png")  # Save the plot as an image
plt.show()


######################################################################################################################


import os
import random
import nltk
from PIL import Image, ImageDraw, ImageFont

# Ensure nltk resources are available
nltk.download('words')
from nltk.corpus import words

# Parameters
OUTPUT_DIR = "ocr_dataset"
IMAGE_SIZE = (256, 64)
FONT_SIZE = 32
NUM_SAMPLES = 100000

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fetch word list and sample words
word_list = words.words()
sampled_words = random.sample(word_list, NUM_SAMPLES)

# Load a default font
try:
    font = ImageFont.truetype("arial.ttf", FONT_SIZE)
except IOError:
    font = ImageFont.load_default()

# Function to sanitize words for filenames
def sanitize_word(word):
    return word.replace("/", "").replace("\\", "")

# Function to generate and save an image for a word
def create_image_for_word(word, font, output_dir, image_size):
    sanitized_word = sanitize_word(word)
    img = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(img)

    # Calculate text position to center it
    text_bbox = draw.textbbox((0, 0), sanitized_word, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

    # Draw the text and save the image
    draw.text(position, sanitized_word, fill="black", font=font)
    img.save(os.path.join(output_dir, f"{sanitized_word}.png"))

# Generate images for sampled words
for word in sampled_words:
    create_image_for_word(word, font, OUTPUT_DIR, IMAGE_SIZE)

print(f"Dataset generation complete. Images saved to {OUTPUT_DIR}")


import os
import shutil
import random

# Paths
OCR_DATASET_DIR = "../../../../ocr_dataset"
TRAIN_DIR = os.path.join(OCR_DATASET_DIR, "train")
VAL_DIR = os.path.join(OCR_DATASET_DIR, "val")
TEST_DIR = os.path.join(OCR_DATASET_DIR, "test")

# Directories to clean
split_dirs = [TRAIN_DIR, VAL_DIR, TEST_DIR]

# Clean existing split directories if they exist
for dir_path in split_dirs:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

# Get list of all image files
image_filenames = [f for f in os.listdir(OCR_DATASET_DIR) if f.endswith(".png")]

# Shuffle the filenames randomly
random.shuffle(image_filenames)

# Calculate split indices
total_files = len(image_filenames)
train_split = int(0.8 * total_files)
val_split = int(0.9 * total_files)

# Split files
split_files = {
    TRAIN_DIR: image_filenames[:train_split],
    VAL_DIR: image_filenames[train_split:val_split],
    TEST_DIR: image_filenames[val_split:]
}

# Move files to respective directories
for dest_dir, files in split_files.items():
    for file in files:
        src_path = os.path.join(OCR_DATASET_DIR, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.copy2(src_path, dest_path)  # Use copy2 to preserve metadata

# Print statistics
print(f"Total files: {total_files}")
print(f"Training files: {len(split_files[TRAIN_DIR])}")
print(f"Validation files: {len(split_files[VAL_DIR])}")
print(f"Test files: {len(split_files[TEST_DIR])}")

# Display samples from each split
for split_name, split_dir in zip(["Training", "Validation", "Test"], split_dirs):
    sample_files = sorted(os.listdir(split_dir))[:5]  # Get the first 5 files alphabetically
    print(f"\nSample of {split_name} files:", sample_files)


import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class OCRDataset(Dataset):
    def __init__(self, image_dir, char_to_idx, transform=None, max_length=30):
        """
        Initialize the OCR dataset.

        Args:
            image_dir (str): Path to the directory containing images.
            char_to_idx (dict): Mapping of characters to indices.
            transform (callable, optional): Transformation to apply to the images.
            max_length (int): Maximum length of encoded labels (for padding).
        """
        self.image_dir = image_dir
        self.image_files = [file for file in os.listdir(image_dir) if file.endswith(".png")]
        self.char_to_idx = char_to_idx
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding encoded label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (transformed image, padded label tensor)
        """
        # Get the filename and label
        image_name = self.image_files[idx]
        label = os.path.splitext(image_name)[0]  # Extract label by removing the file extension

        # Encode the label using the character-to-index mapping
        label_encoded = [self.char_to_idx[char] for char in label]

        # Load and transform the image
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Create a padded label tensor
        label_tensor = torch.full((self.max_length,), -1, dtype=torch.long)  # Padding token (-1)
        label_tensor[:len(label_encoded)] = torch.tensor(label_encoded)

        return image, label_tensor

# Character set including letters, numbers, and common special characters
CHAR_SET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '-"
char_to_idx = {char: idx for idx, char in enumerate(CHAR_SET)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}



# Create datasets with transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
batch_size = 32
learning_rate = 0.001
num_epochs = 5

# Create datasets
train_dataset = OCRDataset("../../../../ocr_dataset/train", char_to_idx, transform=transform, max_length=30)
val_dataset = OCRDataset("../../../../ocr_dataset/val", char_to_idx, transform=transform, max_length=30)
test_dataset = OCRDataset("../../../../ocr_dataset/test", char_to_idx, transform=transform, max_length=30)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class OCRModel(nn.Module):
    def __init__(self, num_classes, hidden_dim=256):
        """
        OCR Model combining a CNN encoder and an RNN decoder.

        Args:
            num_classes (int): Number of output classes (characters).
            hidden_dim (int): Hidden dimension size for the RNN.
        """
        super(OCRModel, self).__init__()

        # CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # RNN Decoder
        self.rnn = nn.GRU(
            input_size=64 * (256 // 4),  # Feature vector size from CNN
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Output tensor of shape [B, Seq_len, Num_classes].
        """
        batch_size = x.size(0)

        # CNN encoder
        features = self.cnn(x)  # Shape: [B, 64, H/4, W/4]
        features = features.permute(0, 2, 3, 1)  # Shape: [B, H/4, W/4, 64]
        features = features.reshape(batch_size, -1, 64 * (256 // 4))  # Shape: [B, Seq_len, Feature_dim]

        # RNN decoder
        rnn_out, _ = self.rnn(features)  # Shape: [B, Seq_len, Hidden_dim]

        # Fully connected layer
        output = self.fc(rnn_out)  # Shape: [B, Seq_len, Num_classes]

        return output


# Initialize the model, loss function, and optimizer
model = OCRModel(num_classes=len(char_to_idx))

# CrossEntropyLoss with padding token ignored
criterion = nn.CrossEntropyLoss(ignore_index=-1)

# Optimizer
learning_rate = 0.001  # Define learning rate if not already set
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: The OCR model.
        train_loader: DataLoader for the training dataset.
        criterion: Loss function.
        optimizer: Optimizer for backpropagation.
        device: Device to run computations on (CPU or GPU).

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to the specified device
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

        # Match labels to sequence length
        batch_size, seq_len, num_classes = outputs.shape
        labels = labels[:, :seq_len]

        # Compute loss
        loss = 0
        for i in range(seq_len):
            curr_output = outputs[:, i, :]  # Output at time step i
            curr_labels = labels[:, i]     # Labels at time step i

            mask = curr_labels != -1       # Ignore padding tokens
            if mask.any():
                curr_loss = criterion(curr_output[mask], curr_labels[mask])
                loss += curr_loss

        loss /= seq_len  # Normalize by sequence length

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """
    Validate the model on the validation dataset.

    Args:
        model: The OCR model.
        val_loader: DataLoader for the validation dataset.
        criterion: Loss function.
        device: Device to run computations on (CPU or GPU).

    Returns:
        Tuple of average validation loss and character-level accuracy.
    """
    model.eval()
    total_loss = 0
    correct_chars = 0
    total_chars = 0

    with torch.no_grad():
        for images, labels in val_loader:
            # Move data to the specified device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Match labels to sequence length
            batch_size, seq_len, num_classes = outputs.shape
            labels = labels[:, :seq_len]

            # Compute loss and accuracy
            for i in range(seq_len):
                curr_output = outputs[:, i, :]
                curr_labels = labels[:, i]

                mask = curr_labels != -1
                if mask.any():
                    curr_loss = criterion(curr_output[mask], curr_labels[mask])
                    total_loss += curr_loss.item()

                    predictions = curr_output[mask].argmax(dim=1)
                    correct_chars += (predictions == curr_labels[mask]).sum().item()
                    total_chars += mask.sum().item()

    avg_loss = total_loss / (len(val_loader) * seq_len)
    accuracy = correct_chars / total_chars if total_chars > 0 else 0
    return avg_loss, accuracy


def decode_prediction(prediction, idx_to_char):
    """
    Decode a prediction tensor into a string.

    Args:
        prediction: Tensor containing predicted indices.
        idx_to_char: Dictionary mapping indices to characters.

    Returns:
        Decoded string.
    """
    return ''.join(idx_to_char[idx.item()] for idx in prediction if idx != -1 and idx.item() < len(idx_to_char))


# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    # Train for one epoch
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

    # Validate
    val_loss, accuracy = validate(model, val_loader, criterion, device)

    # Display epoch results
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Character Accuracy: {accuracy:.4f}")

    # Print examples every few epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            images, labels = next(iter(val_loader))
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=2)

            # Display example predictions
            for i in range(min(3, len(images))):
                true_text = decode_prediction(labels[i], idx_to_char)
                pred_text = decode_prediction(predictions[i].cpu(), idx_to_char)
                print(f"\nExample {i + 1}:")
                print(f"True: {true_text}")
                print(f"Pred: {pred_text}")

    print("-" * 50)
