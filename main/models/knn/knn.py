import numpy as np

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        self.label_mapping = {}
        self.inverse_label_mapping = {}

    def fit(self, X, y):
        self.X_train = np.array(X)
        
        unique_labels = np.unique(y)
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.inverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}
        
        self.y_train = np.array([self.label_mapping[label] for label in y])

    # def euclidean_distance(self, X):
        # X = np.asarray(X, dtype=np.float64)
        # self.X_train = np.asarray(self.X_train, dtype=np.float64)
        # return np.sqrt(np.sum((X - self.X_train) ** 2, axis=1))
    def euclidean_distance(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.sqrt(((X[:, np.newaxis, :] - self.X_train) ** 2).sum(axis=2))
        
    def manhattan_distance(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.abs(X[:, np.newaxis, :] - self.X_train).sum(axis=2)
       

    # def manhattan_distance(self, X):
       
    #     X = np.asarray(X, dtype=np.float64)
    #     self.X_train = np.asarray(self.X_train, dtype=np.float64)
    #     return np.sum(np.abs(X - self.X_train), axis=1)

    # def cosine_distance(self, X):
    #     dot_product = np.dot(self.X_train, X)
    #     norm_X_train = np.linalg.norm(self.X_train, axis=1)
    #     norm_X = np.linalg.norm(X)
    #     cosine_similarity = dot_product / (norm_X * norm_X_train)
    #     return 1 - cosine_similarity
    def cosine_distance(self, X):
        X = np.asarray(X, dtype=np.float64)
        dot_product = np.dot(X, self.X_train.T)
        norm_X = np.linalg.norm(X, axis=1)[:, np.newaxis]
        norm_X_train = np.linalg.norm(self.X_train, axis=1)
        cosine_similarity = dot_product / (norm_X * norm_X_train)
        return 1 - cosine_similarity

        

    def compute_distance(self, X):
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(X)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(X)
        elif self.distance_metric == 'cosine':
            return self.cosine_distance(X)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def predict(self, X):
     X = np.array(X)
     batch_size = 100
     y_pred = []
     self.count=0
    
     for i in range(0, len(X), batch_size):
        # self.count+=1
        # print(self.count)
        X_batch = X[i:i + batch_size]
        
        distances = self.compute_distance(X_batch)
        k_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_indices]
        # print (k_nearest_labels)

        
        y_batch_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=k_nearest_labels)
        # print(y_batch_pred)
        y_pred.extend(y_batch_pred)

     labels=[self.inverse_label_mapping[label] for label in y_pred]
    #  print(labels)
     return labels
        
    #  return np.array(y_pred)
    
    # def predict(self, X):
    #     X = np.array(X)
    #     y_pred = np.zeros(X.shape[0], dtype=self.y_train.dtype)
    #     # print(self.y_train)
    #     for i, x in enumerate(X):
    #         # self.count += 1
    #         # print("Prediction number ", self.count)
    #         distances = self.compute_distance(x)
    #         k_indices = np.argsort(distances)[:self.k]
    #         k_nearest_labels = self.y_train[k_indices]
    #         # y_pred[i] = self.inverse_label_mapping[np.argmax(np.bincount(k_nearest_labels))]
    #         labels, counts = np.unique(k_nearest_labels, return_counts=True)
    #         y_pred[i] = labels[np.argmax(counts)]
    #         print(y_pred[i])
        
    #     labels=[self.inverse_label_mapping[label] for label in y_pred]
    #     print(labels)
    #     return labels

    # def accuracy(self, X_test, y_test):
    #     predictions = self.predict(X_test)
    #     self.count = 0
    #     return np.mean(predictions == y_test)


# class Metrics:
#     def __init__(self):
#         pass

#     @staticmethod
#     def accuracy(y_true, y_pred):
#         return np.mean(y_true == y_pred)

#     @staticmethod
#     def precision(y_true, y_pred, average='macro'):
#         unique_labels = np.unique(y_true)
#         precisions = []

#         for label in unique_labels:
#             true_positive = np.sum((y_true == label) & (y_pred == label))
#             predicted_positive = np.sum(y_pred == label)
#             precision = true_positive / predicted_positive if predicted_positive > 0 else 0
#             precisions.append(precision)

#         if average == 'macro':
#             return np.mean(precisions)
#         elif average == 'micro':
#             true_positive_total = np.sum(y_true == y_pred)
#             predicted_positive_total = len(y_pred)
#             return true_positive_total / predicted_positive_total if predicted_positive_total > 0 else 0

#     @staticmethod
#     def recall(y_true, y_pred, average='macro'):
#         unique_labels = np.unique(y_true)
#         recalls = []

#         for label in unique_labels:
#             true_positive = np.sum((y_true == label) & (y_pred == label))
#             actual_positive = np.sum(y_true == label)
#             recall = true_positive / actual_positive if actual_positive > 0 else 0
#             recalls.append(recall)

#         if average == 'macro':
#             return np.mean(recalls)
#         elif average == 'micro':
#             true_positive_total = np.sum(y_true == y_pred)
#             actual_positive_total = len(y_true)
#             return true_positive_total / actual_positive_total if actual_positive_total > 0 else 0

#     @staticmethod
#     def f1_score(y_true, y_pred, average='macro'):
#         precision = Metrics.precision(y_true, y_pred, average=average)
#         recall = Metrics.recall(y_true, y_pred, average=average)
        
#         if precision + recall == 0:
#             return 0
#         return 2 * (precision * recall) / (precision + recall)

class Metrics:
    def __init__(self, label_mapping=None):
        self.label_mapping = label_mapping

    def _map_labels(self, y_true, y_pred):
        if self.label_mapping is not None:
            inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
            y_true = np.array([self.label_mapping[label] for label in y_true])
            y_pred = np.array([self.label_mapping[label] for label in y_pred])
        return y_true, y_pred

    def accuracy(self, y_true, y_pred):
        y_true, y_pred = self._map_labels(y_true, y_pred)
        return np.mean(y_true == y_pred)

    def precision(self, y_true, y_pred, average='macro'):
        y_true, y_pred = self._map_labels(y_true, y_pred)
        unique_labels = np.unique(y_true)
        precisions = []

        for label in unique_labels:
            true_positive = np.sum((y_true == label) & (y_pred == label))
            predicted_positive = np.sum(y_pred == label)
            precision = true_positive / predicted_positive if predicted_positive > 0 else 0
            precisions.append(precision)

        if average == 'macro':
            return np.mean(precisions)
        elif average == 'micro':
            true_positive_total = np.sum(y_true == y_pred)
            predicted_positive_total = len(y_pred)
            return true_positive_total / predicted_positive_total if predicted_positive_total > 0 else 0

    def recall(self, y_true, y_pred, average='macro'):
        y_true, y_pred = self._map_labels(y_true, y_pred)
        unique_labels = np.unique(y_true)
        recalls = []

        for label in unique_labels:
            true_positive = np.sum((y_true == label) & (y_pred == label))
            actual_positive = np.sum(y_true == label)
            recall = true_positive / actual_positive if actual_positive > 0 else 0
            recalls.append(recall)

        if average == 'macro':
            return np.mean(recalls)
        elif average == 'micro':
            true_positive_total = np.sum(y_true == y_pred)
            actual_positive_total = len(y_true)
            return true_positive_total / actual_positive_total if actual_positive_total > 0 else 0

    def f1_score(self, y_true, y_pred, average='macro'):
        precision = self.precision(y_true, y_pred, average=average)
        recall = self.recall(y_true, y_pred, average=average)

        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

