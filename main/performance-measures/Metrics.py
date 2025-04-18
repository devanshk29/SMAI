import numpy as np

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

