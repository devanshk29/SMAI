import numpy as np
import matplotlib.pyplot as plt


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



