
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, hamming_loss
import wandb

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def linear(x):
    return x

def linear_derivative(x):
    return 1

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss functions
def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true

def binary_cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true

def mean_squared_error(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

class UnifiedMLP:
    def __init__(self, input_size, hidden_layers, output_size, task="classification", lr=0.01, epochs=100, 
                 batch_size=32, activation="relu", optimizer="sgd", use_wandb=False):
        """
        Unified MLP class for Classification, Regression, and Multi-Label Classification.
        :param input_size: Number of input features
        :param hidden_layers: List of neurons in each hidden layer
        :param output_size: Number of output neurons
        :param task: "classification", "regression", or "multi_label"
        :param lr: Learning rate for optimization
        :param epochs: Number of training epochs
        :param batch_size: Size of each batch during training
        :param activation: Activation function ('sigmoid', 'tanh', 'relu', or 'linear')
        :param optimizer: Optimization method ('sgd' by default)
        :param use_wandb: Whether to use Weights & Biases for logging
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.task = task
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.optimizer = optimizer
        self.use_wandb = use_wandb
        self.weights = []
        self.biases = []
        self.train_loss_history = []
        self.val_loss_history = []
        self._initialize_weights()

    def _initialize_weights(self):
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def _activate(self, x):
        if self.activation == "sigmoid":
            return sigmoid(x)
        elif self.activation == "tanh":
            return tanh(x)
        elif self.activation == "relu":
            return relu(x)
        else:
            return linear(x)

    def _activate_derivative(self, x):
        if self.activation == "sigmoid":
            return sigmoid_derivative(x)
        elif self.activation == "tanh":
            return tanh_derivative(x)
        elif self.activation == "relu":
            return relu_derivative(x)
        else:
            return linear_derivative(x)

    def _forward(self, X):
        self.layer_inputs = []
        self.layer_outputs = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            self.layer_inputs.append(z)
            self.layer_outputs.append(self._activate(z))

        z = np.dot(self.layer_outputs[-1], self.weights[-1]) + self.biases[-1]
        self.layer_inputs.append(z)

        if self.task == "classification":
            self.layer_outputs.append(softmax(z))
        elif self.task == "multi_label":
            self.layer_outputs.append(sigmoid(z))
        else:  # regression
            self.layer_outputs.append(z)

        return self.layer_outputs[-1]

    def _backward(self, X, y):
        gradients_w = [0] * len(self.weights)
        gradients_b = [0] * len(self.biases)

        if self.task == "classification":
            output_error = cross_entropy_derivative(y, self.layer_outputs[-1])
        elif self.task == "multi_label":
            output_error = binary_cross_entropy_derivative(y, self.layer_outputs[-1])
        else:  # regression
            output_error = self.layer_outputs[-1] - y

        for i in reversed(range(len(self.weights))):
            delta = output_error
            gradients_w[i] = np.dot(self.layer_outputs[i].T, delta)
            gradients_b[i] = np.sum(delta, axis=0, keepdims=True)
            if i != 0:
                output_error = np.dot(delta, self.weights[i].T) * self._activate_derivative(self.layer_inputs[i - 1])

        return gradients_w, gradients_b

    def _update_weights(self, gradients_w, gradients_b):
        clip_value = 5  # Gradient clipping
        for i in range(len(self.weights)):
            gradients_w[i] = np.clip(gradients_w[i], -clip_value, clip_value)
            gradients_b[i] = np.clip(gradients_b[i], -clip_value, clip_value)
            self.weights[i] -= self.lr * gradients_w[i]
            self.biases[i] -= self.lr * gradients_b[i]

    def fit(self, X, y, X_val=None, y_val=None):
     if self.use_wandb:
        wandb.init(project="unified-mlp", config={
            "task": self.task,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "optimizer": self.optimizer,
            "learning_rate": self.lr,
            "batch_size": self.batch_size
        })

     for epoch in range(self.epochs):
        for start in range(0, len(X), self.batch_size):
            end = start + self.batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]

            self._forward(X_batch)
            gradients_w, gradients_b = self._backward(X_batch, y_batch)
            self._update_weights(gradients_w, gradients_b)

        # Calculate training loss
        if self.task == "classification":
            train_loss = cross_entropy_loss(y, self._forward(X))
        elif self.task == "multi_label":
            train_loss = binary_cross_entropy_loss(y, self._forward(X))
        else:  # regression
            train_loss = mean_squared_error(y, self._forward(X))

        self.train_loss_history.append(train_loss)

        # Calculate validation loss if validation data is provided
        if X_val is not None and y_val is not None:
            if self.task == "classification":
                val_loss = cross_entropy_loss(y_val, self._forward(X_val))
            elif self.task == "multi_label":
                val_loss = binary_cross_entropy_loss(y_val, self._forward(X_val))
            else:  # regression
                val_loss = mean_squared_error(y_val, self._forward(X_val))
        else:
            val_loss = None
        
        self.val_loss_history.append(val_loss)

        if self.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss if val_loss is not None else 'N/A'
            })

        print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss}, Val Loss: {val_loss if val_loss else 'N/A'}")


    def predict(self, X, threshold=0.5):
        output = self._forward(X)
        if self.task == "classification":
            return np.argmax(output, axis=1)
        elif self.task == "multi_label":
            return (output > threshold).astype(int)
        else:  # regression
            return output.flatten()

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        
        if self.task == "classification":
            accuracy = np.mean(y_pred == np.argmax(y_true, axis=1)) * 100
            return {"accuracy": accuracy}
        elif self.task == "multi_label":
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='samples')
            recall = recall_score(y_true, y_pred, average='samples')
            f1 = f1_score(y_true, y_pred, average='samples')
            h_loss = hamming_loss(y_true, y_pred)
            return {
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'hamming_loss': h_loss
            }
        else:  # regression
            mse = mean_squared_error(y_true, y_pred)
            return {"mean_squared_error": mse}

    def gradient_check(self, X, y, epsilon=1e-7, num_gradients=10):
        numerical_grads = []
        count = 0

        for i in range(len(self.weights)):
            w_shape = self.weights[i].shape
            for j in range(w_shape[0]):
                for k in range(w_shape[1]):
                    if count >= num_gradients:
                        return numerical_grads

                    original_value = self.weights[i][j, k]

                    self.weights[i][j, k] = original_value + epsilon
                    loss_plus = self._calculate_loss(X, y)

                    self.weights[i][j, k] = original_value - epsilon
                    loss_minus = self._calculate_loss(X, y)

                    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
                    numerical_grads.append(numerical_grad)

                    self.weights[i][j, k] = original_value

                    count += 1

        return numerical_grads
        

    def _calculate_loss(self, X, y):
        output = self._forward(X)
        if self.task == "classification":
            return cross_entropy_loss(y, output)
        elif self.task == "multi_label":
            return binary_cross_entropy_loss(y, output)
        else:  # regression
            return mean_squared_error(y, output)
    
    def get_backpropagation_gradients(self, X, y, num_gradients=10):
        """
        Extract the first 'num_gradients' backpropagation gradients.
        """
        gradients_w, _ = self._backward(X, y)
        backprop_grads = []
        count = 0

        for grad in gradients_w:
            for i in range(grad.shape[0]):
                for j in range(grad.shape[1]):
                    if count >= num_gradients:
                        return backprop_grads
                    backprop_grads.append(grad[i, j])
                    count += 1

        return backprop_grads

    def accuracy(self, X, y_true):
        """
        Calculates accuracy based on predicted labels and true labels.
        """
        y_pred = self.predict(X)
        
        if self.task == "classification":
            if len(y_true.shape) > 1:
                y_true = np.argmax(y_true, axis=1)
            accuracy = np.mean(y_pred == y_true)
        elif self.task == "multi_label":
            accuracy = accuracy_score(y_true, y_pred)
        else:  # regression
            raise ValueError("Accuracy is not applicable for regression tasks")
        
        return accuracy * 100
    

    def update_weights(self, dW, dB):
   
        if self.optimizer == 'sgd':
            self.update_sgd(dW, dB)
        elif self.optimizer == 'batch':
            self.update_batch(dW, dB)
        elif self.optimizer == 'mini_batch':
            self.update_mini_batch(dW, dB)
        elif self.optimizer == 'batch_grad':
            self.update_batch_grad(dW, dB)
        else:
            raise ValueError(f"Optimizer {self.optimizer} is not recognized.")

# Stochastic Gradient Descent (SGD)
def update_sgd(self, dW, dB):
    """
    Updates weights using Stochastic Gradient Descent (SGD).
    :param dW: Gradients of the weights
    :param dB: Gradients of the biases
    """
    for i in range(len(self.weights)):
        self.weights[i] -= self.lr * dW[i]
        self.biases[i] -= self.lr * dB[i]

# Batch Gradient Descent
def update_batch(self, dW, dB):
    """
    Updates weights using Batch Gradient Descent.
    :param dW: Gradients of the weights
    :param dB: Gradients of the biases
    """
    for i in range(len(self.weights)):
        self.weights[i] -= self.lr * np.mean(dW[i], axis=0, keepdims=True)
        self.biases[i] -= self.lr * np.mean(dB[i], axis=0, keepdims=True)

# Mini-Batch Gradient Descent
def update_mini_batch(self, dW, dB):
    """
    Updates weights using Mini-Batch Gradient Descent.
    :param dW: Gradients of the weights
    :param dB: Gradients of the biases
    """
    batch_size = self.batch_size
    for i in range(len(self.weights)):
        self.weights[i] -= self.lr * np.mean(dW[i][:batch_size], axis=0, keepdims=True)
        self.biases[i] -= self.lr * np.mean(dB[i][:batch_size], axis=0, keepdims=True)

# Batch Gradient with Momentum (Momentum-based Batch Gradient Descent)
def update_batch_grad(self, dW, dB, beta=0.9):
    """
    Updates weights using Batch Gradient Descent with Momentum.
    :param dW: Gradients of the weights
    :param dB: Gradients of the biases
    :param beta: Momentum coefficient (default 0.9)
    """
    if not hasattr(self, 'velocity_w'):
        # Initialize velocity terms if they don't exist
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]

    for i in range(len(self.weights)):
        self.velocity_w[i] = beta * self.velocity_w[i] + (1 - beta) * dW[i]
        self.velocity_b[i] = beta * self.velocity_b[i] + (1 - beta) * dB[i]

        self.weights[i] -= self.lr * self.velocity_w[i]
        self.biases[i] -= self.lr * self.velocity_b[i]






# import numpy as np

# # Activation functions
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     return sigmoid(x) * (1 - sigmoid(x))

# def tanh(x):
#     return np.tanh(x)

# def tanh_derivative(x):
#     return 1 - np.tanh(x)**2

# def relu(x):
#     return np.maximum(0, x)

# def relu_derivative(x):
#     return np.where(x > 0, 1, 0)

# def linear(x):
#     return x

# def linear_derivative(x):
#     return 1

# def softmax(x):
#     exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
#     return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# def cross_entropy_loss(y_true, y_pred):
#     # Clip predictions to avoid log(0)
#     y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
#     return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# def cross_entropy_derivative(y_true, y_pred):
#     return y_pred - y_true  # Simplified derivative for softmax + cross-entropy

# class MLPClassifier:
#     def __init__(self, input_size, hidden_layers, output_size, lr=0.01, epochs=100, batch_size=32, activation="relu", optimizer="sgd"):
#         self.input_size = input_size
#         self.hidden_layers = hidden_layers
#         self.output_size = output_size
#         self.lr = lr
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.activation = activation
#         self.optimizer = optimizer

#         # Initialize weights and biases for each layer
#         self.weights = []
#         self.biases = []
#         self._initialize_weights()

#     def _initialize_weights(self):
#         # Xavier initialization
#         layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
#         for i in range(len(layer_sizes) - 1):
#             self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i]))
#             self.biases.append(np.zeros((1, layer_sizes[i+1])))

#     def _activate(self, x):
#         if self.activation == "sigmoid":
#             return sigmoid(x)
#         elif self.activation == "tanh":
#             return tanh(x)
#         elif self.activation == "relu":
#             return relu(x)
#         else:  # Linear
#             return linear(x)

#     def _activate_derivative(self, x):
#         if self.activation == "sigmoid":
#             return sigmoid_derivative(x)
#         elif self.activation == "tanh":
#             return tanh_derivative(x)
#         elif self.activation == "relu":
#             return relu_derivative(x)
#         else:  # Linear
#             return linear_derivative(x)

#     def _forward(self, X):
#         self.layer_inputs = []
#         self.layer_outputs = [X]
#         for i in range(len(self.weights) - 1):
#             z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
#             self.layer_inputs.append(z)
#             self.layer_outputs.append(self._activate(z))

#         # Output layer with softmax
#         z = np.dot(self.layer_outputs[-1], self.weights[-1]) + self.biases[-1]
#         self.layer_inputs.append(z)
#         self.layer_outputs.append(softmax(z)) 
#         return self.layer_outputs[-1]

#     def _backward(self, X, y):
#         gradients_w = [0] * len(self.weights)
#         gradients_b = [0] * len(self.biases)
#         # print(np.sum(np.sum(self.layer_outputs[-1]>0.5, axis=1)))
#         assert(np.sum(np.sum(self.layer_outputs[-1], axis=1)))
#         # Cross-entropy loss derivative (output layer)
#         output_error = cross_entropy_derivative(y, self.layer_outputs[-1])
#         for i in reversed(range(len(self.weights))):
#             delta = output_error
#             gradients_w[i] = np.dot(self.layer_outputs[i].T, delta)
#             gradients_b[i] = np.sum(delta, axis=0, keepdims=True)

#             if i != 0:
#                 output_error = np.dot(delta, self.weights[i].T) * self._activate_derivative(self.layer_inputs[i - 1])

#         return gradients_w, gradients_b

#     def _update_weights(self, gradients_w, gradients_b):
#         for i in range(len(self.weights)):
#             self.weights[i] -= self.lr * gradients_w[i]
#             self.biases[i] -= self.lr * gradients_b[i]

#     def fit(self, X, y):
#         for epoch in range(self.epochs):
#             # Split into batches
#             for start in range(0, len(X), self.batch_size):
#                 end = start + self.batch_size
#                 X_batch = X[start:end]
#                 y_batch = y[start:end]

#                 # Forward pass
#                 output = self._forward(X_batch)

#                 # Backward pass
#                 gradients_w, gradients_b = self._backward(X_batch, y_batch)

#                 # Update weights
#                 self._update_weights(gradients_w, gradients_b)

#             # Optionally, calculate and print the loss for monitoring
#             loss = cross_entropy_loss(y, self._forward(X))
#             # print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss}")

#     def predict(self, X):
#         output = self._forward(X)
#         # print((output > 0.5).shape)
#         output_array = []
#         for i in range(len(output)):
#             output_array.append(np.argmax(output[i]))
#         return np.array(output_array)

#     def gradient_check(self, X, y, epsilon=1e-7, num_gradients=10):
#         """
#         Compute the first 'num_gradients' numerical gradients using finite difference method.
#         """
#         numerical_grads = []
#         count = 0

#         for i in range(len(self.weights)):
#             w_shape = self.weights[i].shape
#             for j in range(w_shape[0]):
#                 for k in range(w_shape[1]):
#                     if count >= num_gradients:
#                         break

#                     # Evaluate gradients numerically
#                     original_value = self.weights[i][j, k]

#                     # Positive perturbation
#                     self.weights[i][j, k] = original_value + epsilon
#                     loss_plus = cross_entropy_loss(y, self._forward(X))

#                     # Negative perturbation
#                     self.weights[i][j, k] = original_value - epsilon
#                     loss_minus = cross_entropy_loss(y, self._forward(X))

#                     # Approximate gradient
#                     numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
#                     numerical_grads.append(numerical_grad)

#                     # Restore original value
#                     self.weights[i][j, k] = original_value

#                     count += 1

#                 if count >= num_gradients:
#                     break

#         return numerical_grads

#     def get_backpropagation_gradients(self, X, y, num_gradients=10):
#         """
#         Extract the first 'num_gradients' backpropagation gradients.
#         """
#         gradients_w, _ = self._backward(X, y)
#         backprop_grads = []
#         count = 0

#         for grad in gradients_w:
#             for i in range(grad.shape[0]):
#                 for j in range(grad.shape[1]):
#                     if count >= num_gradients:
#                         return backprop_grads
#                     backprop_grads.append(grad[i, j])
#                     count += 1

#         return backprop_grads

#     def accuracy(self, X, y_true):
#         """
#         Calculates accuracy based on predicted labels and true labels.

#         Parameters:
#         - X: The input features to predict on.
#         - y_true: The true labels (not one-hot encoded).

#         Returns:
#         - accuracy: The percentage of correct predictions.
#         """
#         # Get the predicted labels
#         y_pred = self.predict(X)+3

#         # Check if true labels are one-hot encoded, convert to label index
#         if len(y_true.shape) > 1:
#             y_true = np.argmax(y_true, axis=1)

#         # Calculate accuracy as the percentage of correct predictions
#         accuracy = np.mean(y_pred == y_true)
#         return accuracy * 100





# import numpy as np

# # Activation functions
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     return sigmoid(x) * (1 - sigmoid(x))

# def tanh(x):
#     return np.tanh(x)

# def tanh_derivative(x):
#     return 1 - np.tanh(x)**2

# def relu(x):
#     return np.maximum(0, x)

# def relu_derivative(x):
#     return np.where(x > 0, 1, 0)

# # Loss functions
# def mse_loss(y_true, y_pred):
#     return np.mean((y_true - y_pred) ** 2)

# def bce_loss(y_true, y_pred):
#     # Adding a small epsilon to avoid log(0) error
#     epsilon = 1e-7
#     y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
#     return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# # MLP for Regression
# class MLPRegressor:
#     def __init__(self, input_size, hidden_layers, output_size, lr=0.01, epochs=100, batch_size=32, activation="sigmoid", optimizer="sgd", loss_function="mse"):
#         self.input_size = input_size
#         self.hidden_layers = hidden_layers
#         self.output_size = output_size
#         self.lr = lr
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.activation = activation
#         self.optimizer = optimizer
#         self.loss_function = loss_function
#         self.loss_history = []

#         # Initialize weights and biases for each layer
#         self.weights = []
#         self.biases = []
#         self._initialize_weights()

#     def _initialize_weights(self):
#         # Xavier initialization
#         layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
#         for i in range(len(layer_sizes) - 1):
#             self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i]))
#             self.biases.append(np.zeros((1, layer_sizes[i+1])))

#     def _activate(self, x):
#         if self.activation == "sigmoid":
#             return sigmoid(x)
#         elif self.activation == "tanh":
#             return tanh(x)
#         elif self.activation == "relu":
#             return relu(x)
#         else:
#             return x  # Linear activation for regression

#     def _activate_derivative(self, x):
#         if self.activation == "sigmoid":
#             return sigmoid_derivative(x)
#         elif self.activation == "tanh":
#             return tanh_derivative(x)
#         elif self.activation == "relu":
#             return relu_derivative(x)
#         else:
#             return 1  # Linear derivative for regression

#     def _forward(self, X):
#         self.layer_inputs = []
#         self.layer_outputs = [X]
#         for i in range(len(self.weights) - 1):
#             z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
#             self.layer_inputs.append(z)
#             self.layer_outputs.append(self._activate(z))

#         # Output layer (no activation for regression, sigmoid for classification)
#         z = np.dot(self.layer_outputs[-1], self.weights[-1]) + self.biases[-1]
#         self.layer_inputs.append(z)

#         if self.loss_function == "bce":
#             self.layer_outputs.append(sigmoid(z))  # Sigmoid for BCE
#         else:
#             self.layer_outputs.append(z)  # Linear output for MSE

#         return self.layer_outputs[-1]

#     def _backward(self, X, y):
#         gradients_w = [0] * len(self.weights)
#         gradients_b = [0] * len(self.biases)

#         # Loss derivative (either MSE or BCE)
#         if self.loss_function == "mse":
#             output_error = self.layer_outputs[-1] - y
#         elif self.loss_function == "bce":
#             output_error = (self.layer_outputs[-1] - y) / (self.layer_outputs[-1] * (1 - self.layer_outputs[-1]))

#         for i in reversed(range(len(self.weights))):
#             # Calculate gradients for weights and biases
#             delta = output_error
#             gradients_w[i] = np.dot(self.layer_outputs[i].T, delta)
#             gradients_b[i] = np.sum(delta, axis=0, keepdims=True)

#             if i != 0:
#                 # Backpropagate the error
#                 output_error = np.dot(delta, self.weights[i].T)
#                 output_error *= self._activate_derivative(self.layer_inputs[i - 1])  # Element-wise multiplication

#         return gradients_w, gradients_b

#     def _update_weights(self, gradients_w, gradients_b):
#         for i in range(len(self.weights)):
#             self.weights[i] -= self.lr * gradients_w[i]
#             self.biases[i] -= self.lr * gradients_b[i]

#     def fit(self, X, y):
#         for epoch in range(self.epochs):
#             for start in range(0, len(X), self.batch_size):
#                 end = start + self.batch_size
#                 X_batch = X[start:end]
#                 y_batch = y[start:end]

#                 # Forward pass
#                 self._forward(X_batch)

#                 # Backward pass
#                 gradients_w, gradients_b = self._backward(X_batch, y_batch)

#                 # Update weights
#                 self._update_weights(gradients_w, gradients_b)

#             # Calculate and log the loss
#             if self.loss_function == "mse":
#                 loss = mse_loss(y, self._forward(X))
#             elif self.loss_function == "bce":
#                 loss = bce_loss(y, self._forward(X))

#             self.loss_history.append(loss)
#             print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss}")

#     def predict(self, X):
#         return self._forward(X).flatten()







# # import numpy as np

# # # Activation functions
# # def sigmoid(x):
# #     return 1 / (1 + np.exp(-x))

# # def sigmoid_derivative(x):
# #     return sigmoid(x) * (1 - sigmoid(x))

# # def tanh(x):
# #     return np.tanh(x)

# # def tanh_derivative(x):
# #     return 1 - np.tanh(x)**2

# # def relu(x):
# #     return np.maximum(0, x)

# # def relu_derivative(x):
# #     return np.where(x > 0, 1, 0)

# # # Loss functions
# # def mse_loss(y_true, y_pred):
# #     return np.mean((y_true - y_pred) ** 2)

# # def mse_derivative(y_true, y_pred):
# #     return 2 * (y_pred - y_true) / y_true.size

# # def bce_loss(y_true, y_pred):
# #     epsilon = 1e-12  # to avoid log(0)
# #     y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # to prevent log(0)
# #     return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# # def bce_derivative(y_true, y_pred):
# #     epsilon = 1e-12
# #     y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
# #     return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)

# # # MLP for Regression or Classification
# # class MLPRegressor:
# #     def __init__(self, input_size, hidden_layers, output_size, lr=0.01, epochs=100, batch_size=32, 
# #                  activation="sigmoid", optimizer="sgd", loss_function="mse"):
# #         self.input_size = input_size
# #         self.hidden_layers = hidden_layers
# #         self.output_size = output_size
# #         self.lr = lr
# #         self.epochs = epochs
# #         self.batch_size = batch_size
# #         self.activation = activation
# #         self.optimizer = optimizer
# #         self.loss_function = loss_function

# #         # Initialize weights and biases for each layer
# #         self.weights = []
# #         self.biases = []
# #         self._initialize_weights()

# #     def _initialize_weights(self):
# #         # Xavier initialization
# #         layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
# #         for i in range(len(layer_sizes) - 1):
# #             self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i]))
# #             self.biases.append(np.zeros((1, layer_sizes[i+1])))

# #     def _activate(self, x):
# #         if self.activation == "sigmoid":
# #             return sigmoid(x)
# #         elif self.activation == "tanh":
# #             return tanh(x)
# #         elif self.activation == "relu":
# #             return relu(x)
# #         else:
# #             return x  # Linear activation for regression

# #     def _activate_derivative(self, x):
# #         if self.activation == "sigmoid":
# #             return sigmoid_derivative(x)
# #         elif self.activation == "tanh":
# #             return tanh_derivative(x)
# #         elif self.activation == "relu":
# #             return relu_derivative(x)
# #         else:
# #             return 1  # Linear derivative for regression

# #     def _forward(self, X):
# #         self.layer_inputs = []
# #         self.layer_outputs = [X]
# #         for i in range(len(self.weights) - 1):
# #             z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
# #             self.layer_inputs.append(z)
# #             self.layer_outputs.append(self._activate(z))

# #         # Output layer (no activation for regression, sigmoid for BCE loss)
# #         z = np.dot(self.layer_outputs[-1], self.weights[-1]) + self.biases[-1]
# #         if self.loss_function == "bce":
# #             self.layer_inputs.append(z)
# #             self.layer_outputs.append(sigmoid(z))
# #         else:
# #             self.layer_inputs.append(z)
# #             self.layer_outputs.append(z)
# #         return self.layer_outputs[-1]

# #     def _backward(self, X, y):
# #         gradients_w = [0] * len(self.weights)
# #         gradients_b = [0] * len(self.biases)

# #         # Loss derivative based on the selected loss function
# #         if self.loss_function == "mse":
# #             output_error = mse_derivative(y, self.layer_outputs[-1])
# #         elif self.loss_function == "bce":
# #             output_error = bce_derivative(y, self.layer_outputs[-1])

# #         # Backpropagation
# #         for i in reversed(range(len(self.weights))):
# #             # Calculate gradients for weights and biases
# #             delta = output_error
# #             gradients_w[i] = np.dot(self.layer_outputs[i].T, delta)
# #             gradients_b[i] = np.sum(delta, axis=0, keepdims=True)

# #             if i != 0:
# #                 # Backpropagate the error
# #                 output_error = np.dot(delta, self.weights[i].T)
# #                 output_error *= self._activate_derivative(self.layer_inputs[i - 1])

# #         return gradients_w, gradients_b

# #     def _update_weights(self, gradients_w, gradients_b):
# #         for i in range(len(self.weights)):
# #             self.weights[i] -= self.lr * gradients_w[i]
# #             self.biases[i] -= self.lr * gradients_b[i]

# #     def fit(self, X, y):
# #         for epoch in range(self.epochs):
# #             for start in range(0, len(X), self.batch_size):
# #                 end = start + self.batch_size
# #                 X_batch = X[start:end]
# #                 y_batch = y[start:end]

# #                 # Forward pass
# #                 self._forward(X_batch)

# #                 # Backward pass
# #                 gradients_w, gradients_b = self._backward(X_batch, y_batch)

# #                 # Update weights
# #                 self._update_weights(gradients_w, gradients_b)

# #             # Optionally, calculate and print the loss for monitoring
# #             if self.loss_function == "mse":
# #                 loss = mse_loss(y, self._forward(X))
# #             elif self.loss_function == "bce":
# #                 loss = bce_loss(y, self._forward(X))
# #             print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss}")

# #     def predict(self, X):
# #         return self._forward(X).flatten()





import numpy as np
# import wandb

# # Activation functions
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     return sigmoid(x) * (1 - sigmoid(x))

# def tanh(x):
#     return np.tanh(x)

# def tanh_derivative(x):
#     return 1 - np.tanh(x)**2

# def relu(x):
#     return np.maximum(0, x)

# def relu_derivative(x):
#     return np.where(x > 0, 1, 0)

# def linear(x):
#     return x

# def linear_derivative(x):
#     return 1

# def softmax(x):
#     exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
#     return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# def cross_entropy_loss(y_true, y_pred):
#     y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
#     return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# def cross_entropy_derivative(y_true, y_pred):
#     return y_pred - y_true

# class MLPClassifier:
#     def __init__(self, input_size, hidden_layers, output_size, lr=0.01, epochs=100, batch_size=32, activation="relu", optimizer="sgd"):
#         self.input_size = input_size
#         self.hidden_layers = hidden_layers
#         self.output_size = output_size
#         self.lr = lr
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.activation = activation
#         self.optimizer = optimizer
#         self.weights = []
#         self.biases = [] 
#         self.train_loss_history = []
#         self.val_loss_history = []
#         self.train_acc_history = []
#         self.val_acc_history = []
#         self._initialize_weights()

#     def _initialize_weights(self):
#         layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
#         for i in range(len(layer_sizes) - 1):
#             self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i]))
#             self.biases.append(np.zeros((1, layer_sizes[i+1])))

#     def _activate(self, x):
#         if self.activation == "sigmoid":
#             return sigmoid(x)
#         elif self.activation == "tanh":
#             return tanh(x)
#         elif self.activation == "relu":
#             return relu(x)
#         else:
#             return linear(x)

#     def _activate_derivative(self, x):
#         if self.activation == "sigmoid":
#             return sigmoid_derivative(x)
#         elif self.activation == "tanh":
#             return tanh_derivative(x)
#         elif self.activation == "relu":
#             return relu_derivative(x)
#         else:
#             return linear_derivative(x)

#     def _forward(self, X):
#         self.layer_inputs = []
#         self.layer_outputs = [X]
#         for i in range(len(self.weights) - 1):
#             z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
#             self.layer_inputs.append(z)
#             self.layer_outputs.append(self._activate(z))
#         z = np.dot(self.layer_outputs[-1], self.weights[-1]) + self.biases[-1]
#         self.layer_inputs.append(z)
#         self.layer_outputs.append(softmax(z)) 
#         return self.layer_outputs[-1]

#     def _backward(self, X, y):
#         gradients_w = [0] * len(self.weights)
#         gradients_b = [0] * len(self.biases)
#         output_error = cross_entropy_derivative(y, self.layer_outputs[-1])
#         for i in reversed(range(len(self.weights))):
#             delta = output_error
#             gradients_w[i] = np.dot(self.layer_outputs[i].T, delta)
#             gradients_b[i] = np.sum(delta, axis=0, keepdims=True)
#             if i != 0:
#                 output_error = np.dot(delta, self.weights[i].T) * self._activate_derivative(self.layer_inputs[i - 1])
#         return gradients_w, gradients_b

#     def _update_weights(self, gradients_w, gradients_b):
#         for i in range(len(self.weights)):
#             self.weights[i] -= self.lr * gradients_w[i]
#             self.biases[i] -= self.lr * gradients_b[i]

#     def fit(self, X, y, X_val=None, y_val=None):
#         for epoch in range(self.epochs):
#             for start in range(0, len(X), self.batch_size):
#                 end = start + self.batch_size
#                 X_batch = X[start:end]
#                 y_batch = y[start:end]
#                 output = self._forward(X_batch)
#                 gradients_w, gradients_b = self._backward(X_batch, y_batch)
#                 self._update_weights(gradients_w, gradients_b)

#             # Calculate losses and accuracy
#             train_loss = cross_entropy_loss(y, self._forward(X))
#             val_loss = cross_entropy_loss(y_val, self._forward(X_val)) if X_val is not None else None
#             train_accuracy = self.accuracy(X, y)
#             val_accuracy = self.accuracy(X_val, y_val) if X_val is not None else None

#             # Store history for graph plotting
#             self.train_loss_history.append(train_loss)
#             self.val_loss_history.append(val_loss)
#             self.train_acc_history.append(train_accuracy)
#             self.val_acc_history.append(val_accuracy)

#             # Log metrics to W&B
#             wandb.log({
#                 "epoch": epoch,
#                 "train_loss": train_loss,
#                 "val_loss": val_loss,
#                 "train_accuracy": train_accuracy,
#                 "val_accuracy": val_accuracy
#             })

#     def predict(self, X):
#         output = self._forward(X)
#         return np.array([np.argmax(o) for o in output])

#     def accuracy(self, X, y_true):
#         y_pred = self.predict(X)
#         if len(y_true.shape) > 1:
#             y_true = np.argmax(y_true, axis=1)
#         accuracy = np.mean(y_pred == y_true)
#         return accuracy * 100


import numpy as np
# from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, hamming_loss

# # Activation functions
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     return sigmoid(x) * (1 - sigmoid(x))

# def binary_cross_entropy_loss(y_true, y_pred):
#     y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
#     return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# def binary_cross_entropy_derivative(y_true, y_pred):
#     return y_pred - y_true  # Derivative of binary cross-entropy loss

# class MultiLabelMLPClassifier:
#     def __init__(self, input_size, hidden_layers, output_size, lr=0.01, epochs=100, batch_size=32, activation="relu"):
#         self.input_size = input_size
#         self.hidden_layers = hidden_layers
#         self.output_size = output_size
#         self.lr = lr
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.activation = activation

#         # Initialize weights and biases for each layer
#         self.weights = []
#         self.biases = []
#         self._initialize_weights()

#     def _initialize_weights(self):
#         layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
#         for i in range(len(layer_sizes) - 1):
#             self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]))
#             self.biases.append(np.zeros((1, layer_sizes[i + 1])))

#     def _activate(self, x):
#         if self.activation == "sigmoid":
#             return sigmoid(x)
#         elif self.activation == "relu":
#             return np.maximum(0, x)
#         else:
#             return x  # Linear

#     def _activate_derivative(self, x):
#         if self.activation == "sigmoid":
#             return sigmoid_derivative(x)
#         elif self.activation == "relu":
#             return np.where(x > 0, 1, 0)
#         else:
#             return 1  # Linear

#     def _forward(self, X):
#         self.layer_inputs = []
#         self.layer_outputs = [X]

#         for i in range(len(self.weights) - 1):
#             z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
#             self.layer_inputs.append(z)
#             self.layer_outputs.append(self._activate(z))

#         # Output layer uses sigmoid for multi-label classification
#         z = np.dot(self.layer_outputs[-1], self.weights[-1]) + self.biases[-1]
#         self.layer_inputs.append(z)
#         self.layer_outputs.append(sigmoid(z))
#         print(f"Output shape: {self.layer_outputs[-1].shape}")  # Debugging statement to check output shape
#         return self.layer_outputs[-1]

#     def _backward(self, X, y):
#         gradients_w = [0] * len(self.weights)
#         gradients_b = [0] * len(self.biases)
#         print(len(self.weights))  # Debugging statement to check number of weights
#         print(len(self.biases))  # Debugging statement to check number of biases
#         print(y.shape)  # Debugging statement to check y shape
#         print(self.layer_outputs[-1].shape)  # Debugging statement to check output layer shape
        

#         # Binary cross-entropy loss derivative (output layer)
#         output_error = binary_cross_entropy_derivative(y, self.layer_outputs[-1])
#         print(f"Output error shape: {output_error.shape}")  # Debugging statement to check output error shape
#         for i in reversed(range(len(self.weights))):
#             delta = output_error
#             gradients_w[i] = np.dot(self.layer_outputs[i].T, delta)
#             gradients_b[i] = np.sum(delta, axis=0, keepdims=True)

#             if i != 0:
#                 output_error = np.dot(delta, self.weights[i].T) * self._activate_derivative(self.layer_inputs[i - 1])

#         return gradients_w, gradients_b

#     def _update_weights(self, gradients_w, gradients_b):
#         clip_value = 5  # Example value, can be tuned
#         for i in range(len(self.weights)):
#             gradients_w[i] = np.clip(gradients_w[i], -clip_value, clip_value)
#             gradients_b[i] = np.clip(gradients_b[i], -clip_value, clip_value)
#             self.weights[i] -= self.lr * gradients_w[i]
#             self.biases[i] -= self.lr * gradients_b[i]

#     def fit(self, X, y):
#         for epoch in range(self.epochs):
#             for start in range(0, len(X), self.batch_size):
#                 end = start + self.batch_size
#                 X_batch = X[start:end]
#                 y_batch = y[start:end]
#                 print("hihiii")
#                 print(y.shape)

#                 # Forward pass
#                 output = self._forward(X_batch)
#                 print(f"Output shape: {output.shape}")  # Debugging statement to check output shape
#                 # Backward pass
#                 gradients_w, gradients_b = self._backward(X_batch, y_batch)
#                 print(f"Gradients shape: {gradients_w[0].shape}")  # Debugging statement to check gradients shape

#                 # Update weights
#                 self._update_weights(gradients_w, gradients_b)

#             # Optionally, print the loss for monitoring
#             loss = binary_cross_entropy_loss(y, self._forward(X))
#             print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss}")

#     def predict(self, X, threshold=0.5):
#         output = self._forward(X)
#         print(f"Prediction output shape: {output.shape}")  # Debugging statement to check output shape
#         return (output > threshold).astype(int)

#     def evaluate(self, X, y_true, threshold=0.5):
#         y_pred = self.predict(X, threshold=threshold)
        
#         acc = accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred, average='samples')
#         recall = recall_score(y_true, y_pred, average='samples')
#         f1 = f1_score(y_true, y_pred, average='samples')
#         h_loss = hamming_loss(y_true, y_pred)

#         return {
#             'accuracy': acc,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1,
#             'hamming_loss': h_loss
#         }

