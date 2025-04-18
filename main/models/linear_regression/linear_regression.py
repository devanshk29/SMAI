import numpy as np
import matplotlib.pyplot as plt


def calculate_metrics2(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    variance = np.var(y_pred)
    std_dev = np.std(y_pred)
    return mse, variance, std_dev

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=10000, lambda_=0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.coefficients = np.zeros(n_features)
        self.intercept = 0
        
        for _ in range(self.n_iterations):
            y_pred = self.predict(X)
            
            dw = (1 / n_samples) * (np.dot(X.T, (y_pred - y)) + self.lambda_ * self.coefficients)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.coefficients -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)



class PolynomialRegression:
    def __init__(self, degree, regularization=None, alpha=0.01, learning_rate=0.001, num_iterations=1000):
        self.degree = degree
        self.regularization = regularization
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.coefficients = None
        self.history = []

    def fit(self, X, y,path=None,path2=None):

        if(path!=None):
            self.coefficients = np.load('./assignments/1/best_model_params.npy')
            return
        
        X_poly = self._polynomial_features(X)
        n_samples, n_features = X_poly.shape
        self.coefficients = np.zeros(n_features)

        for iteration in range(self.num_iterations):
            y_pred = self.predict(X)
            error = y_pred - y

            if self.regularization == 'L2':
                gradient = (1/n_samples) * (X_poly.T @ error + self.alpha * self.coefficients)
            elif self.regularization == 'L1':
                gradient = (1/n_samples) * (X_poly.T @ error + self.alpha * np.sign(self.coefficients))
            else:
                gradient = (1/n_samples) * X_poly.T @ error

            self.coefficients -= self.learning_rate * gradient

        
            if path2!=None:
                metrics = calculate_metrics2(y, y_pred)
                self.history.append((iteration, *metrics))
                self.plot_progress(X, y, y_pred, metrics, iteration, path2,self.degree)

    def predict(self, X):
        X_poly = self._polynomial_features(X)
        return X_poly @ self.coefficients

    def _polynomial_features(self, X):
        return np.column_stack([X**i for i in range(self.degree + 1)])
    
    def plot_progress(self, X, y, y_pred, metrics, iteration, save_path,k):
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 2, 1)
        plt.scatter(X, y, color='blue', label='Data')
        plt.plot(X, y_pred, color='red', label='Fit')
        plt.title(f'Fitting Line - Iteration {iteration}')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(iteration, metrics[0], 'bo')
        plt.title(f'MSE: {metrics[0]:.4f}')
        
        plt.subplot(2, 2, 3)
        plt.plot(iteration, metrics[1], 'go')
        plt.title(f'Standard Deviation: {metrics[1]:.4f}')
        
        plt.subplot(2, 2, 4)
        plt.plot(iteration, metrics[2], 'ro')
        plt.title(f'Variance: {metrics[2]:.4f}')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/{k}_iteration_{iteration}.png")
        plt.close()


import numpy as np
import matplotlib.pyplot as plt


def calculate_metrics2(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    variance = np.var(y_pred)
    std_dev = np.std(y_pred)
    return mse, variance, std_dev

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=10000, lambda_=0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.coefficients = np.zeros(n_features)
        self.intercept = 0
        
        for _ in range(self.n_iterations):
            y_pred = self.predict(X)
            
            dw = (1 / n_samples) * (np.dot(X.T, (y_pred - y)) + self.lambda_ * self.coefficients)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.coefficients -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)



class PolynomialRegression:
    def __init__(self, degree, regularization=None, alpha=0.01, learning_rate=0.001, num_iterations=1000):
        self.degree = degree
        self.regularization = regularization
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.coefficients = None
        self.history = []

    def fit(self, X, y,path=None,path2=None):

        if(path!=None):
            self.coefficients = np.load('./assignments/1/best_model_params.npy')
            return
        
        X_poly = self._polynomial_features(X)
        n_samples, n_features = X_poly.shape
        self.coefficients = np.zeros(n_features)

        for iteration in range(self.num_iterations):
            y_pred = self.predict(X)
            error = y_pred - y

            if self.regularization == 'L2':
                gradient = (1/n_samples) * (X_poly.T @ error + self.alpha * self.coefficients)
            elif self.regularization == 'L1':
                gradient = (1/n_samples) * (X_poly.T @ error + self.alpha * np.sign(self.coefficients))
            else:
                gradient = (1/n_samples) * X_poly.T @ error

            self.coefficients -= self.learning_rate * gradient

        
            if path2!=None:
                metrics = calculate_metrics2(y, y_pred)
                self.history.append((iteration, *metrics))
                self.plot_progress(X, y, y_pred, metrics, iteration, path2,self.degree)

    def predict(self, X):
        X_poly = self._polynomial_features(X)
        return X_poly @ self.coefficients

    def _polynomial_features(self, X):
        return np.column_stack([X**i for i in range(self.degree + 1)])
    
    def plot_progress(self, X, y, y_pred, metrics, iteration, save_path,k):
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 2, 1)
        plt.scatter(X, y, color='blue', label='Data')
        plt.plot(X, y_pred, color='red', label='Fit')
        plt.title(f'Fitting Line - Iteration {iteration}')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(iteration, metrics[0], 'bo')
        plt.title(f'MSE: {metrics[0]:.4f}')
        
        plt.subplot(2, 2, 3)
        plt.plot(iteration, metrics[1], 'go')
        plt.title(f'Standard Deviation: {metrics[1]:.4f}')
        
        plt.subplot(2, 2, 4)
        plt.plot(iteration, metrics[2], 'ro')
        plt.title(f'Variance: {metrics[2]:.4f}')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/{k}_iteration_{iteration}.png")
        plt.close()

