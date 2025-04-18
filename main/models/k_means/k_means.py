import numpy as np

class KMeans:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        self.wcss = None

    def _init_centroids(self, X):
        
        centroids = [X[np.random.choice(X.shape[0])]]
        for _ in range(1, self.k):
            distances = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()
            i = np.argmax(cumulative_probabilities >= r)
            centroids.append(X[i])
        return np.array(centroids)

    def fit(self, X):
        self.centroids = self._init_centroids(X)
        
        for _ in range(self.max_iters):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])
            
            if np.all(self.centroids == new_centroids):
                break
            
            self.centroids = new_centroids
        
        self.wcss = sum(np.min(((X - self.centroids[:, np.newaxis])**2).sum(axis=2), axis=0))

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def getCost(self):
        return self.wcss