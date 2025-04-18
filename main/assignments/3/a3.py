import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.MLP.MLP import MLPClassifier
from models.MLP.MLP import UnifiedMLP

# Function to split dataset into training and testing sets
def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    test_size = int(len(indices) * test_size)
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Function to calculate custom accuracy
def custom_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true) * 100

# Function to generate classification report
def custom_classification_report(y_true, y_pred):
    unique_classes = np.unique(y_true)
    report = {}
    
    for cls in unique_classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        report[cls] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': np.sum(y_true == cls)
        }
    
    return report

# Function to generate confusion matrix
def custom_confusion_matrix(y_true, y_pred):
    unique_classes = np.unique(y_true)
    cm = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
    
    for i, true_label in enumerate(y_true):
        pred_label = y_pred[i]
        cm[true_label, pred_label] += 1
    
    return cm

# Load dataset
df = pd.read_csv("./data/external/WineQT.csv")

# Show dataset description
print("Dataset Description:")
print(df.describe())

# Plot distribution of quality labels
plt.figure(figsize=(8, 6))
df['quality'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Wine Quality Labels')
plt.xlabel('Wine Quality')
plt.ylabel('Frequency')
# plt.show()
plt.close()

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Extract features and labels
X = df.drop('quality', axis=1).values
y = df['quality'].values

# Normalize and standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize UnifiedMLP model for gradient checking
mlp_check = UnifiedMLP(input_size=X_scaled.shape[1], hidden_layers=[10, 5], output_size=10, lr=0.01, epochs=100, task="classification")

# Perform gradient checking on a small subset
sample_X = X_scaled[:10]
sample_y = np.eye(10)[y[:10] - y.min()]  # One-hot encoded labels for 10 samples

# Get the first 10 numerical gradients
numerical_grads = mlp_check.gradient_check(sample_X, sample_y, num_gradients=10)

# Get the first 10 backpropagation gradients
backprop_grads = mlp_check.get_backpropagation_gradients(sample_X, sample_y, num_gradients=10)

# Compare numerical and backpropagation gradients side by side
# print("\nComparison of Numerical vs. Backpropagation Gradients:")
# for i, (num_grad, backprop_grad) in enumerate(zip(numerical_grads, backprop_grads)):
#     print(f"Gradient {i+1}: Numerical = {num_grad*10:.6f}, Backpropagation = {backprop_grad:.6f}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = custom_train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Adjust labels to be zero-based
y_min = y.min()
y_train_shifted = y_train - y_min
y_test_shifted = y_test - y_min

# Convert labels to one-hot encoding
num_classes = len(np.unique(y))
y_train_one_hot = np.eye(num_classes)[y_train_shifted]
y_test_one_hot = np.eye(num_classes)[y_test_shifted]

# Initialize UnifiedMLP model for training
mlp = UnifiedMLP(input_size=X_train.shape[1], hidden_layers=[64, 32], output_size=num_classes, 
                 lr=0.01, epochs=100, task="classification", activation="sigmoid")

# Train the MLP
mlp.fit(X_train, y_train_one_hot)

# Make predictions
y_pred = mlp.predict(X_test)

# Calculate custom accuracy
custom_acc = custom_accuracy(y_test_shifted, y_pred)
print(f"Custom Accuracy: {custom_acc:.2f}%")

# Initialize UnifiedMLP model with updated configurations
mlp = UnifiedMLP(input_size=X_train.shape[1], hidden_layers=[64, 16], output_size=num_classes, 
                 lr=0.1, epochs=200, task="classification", activation="tanh", optimizer="sgd", batch_size=32)

# Train the MLP with updated configurations
mlp.fit(X_train, y_train_one_hot)

# Make predictions
y_pred = mlp.predict(X_test)

# Calculate custom accuracy
custom_acc = custom_accuracy(y_test_shifted, y_pred)
print(f"Custom Accuracy with updated configurations: {custom_acc:.2f}%")


############################################################################################################################################################


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import wandb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from models.MLP.MLPwb import MLPClassifier
# from models.MLP.multilableMLP import MultiLabelMLPClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import wandb

# Initialize Weights & Biases
wandb.init(project="mlp-hyperparameter-tuning", name="MLP-Hyperparameter-Tuning")

# Load dataset
df = pd.read_csv("./data/external/WineQT.csv")

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Extract features and labels
X = df.drop('quality', axis=1)
y = df['quality']

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = custom_train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Shift and one-hot encode the labels
y_shifted = y - y.min()
y_train_shifted = y_train - y_train.min()
y_test_shifted = y_test - y_test.min()
num_classes = len(np.unique(y_train_shifted))
y_train_one_hot = np.eye(num_classes)[y_train_shifted]
y_test_one_hot = np.eye(num_classes)[y_test_shifted]

# Define hyperparameters to test
hyperparameters = {
    "learning_rates": [0.001, 0.01, 0.1],
    "hidden_layers": [[64, 32], [128, 64, 32], [32, 16]],
    "activations": ["sigmoid", "tanh", "relu"]
}

# Initialize variables to store the best model
best_f1 = -1
best_hyperparams = None

# # Iterate over all combinations of hyperparameters
# for lr in hyperparameters["learning_rates"]:
#     for hidden_layers in hyperparameters["hidden_layers"]:
#         for activation in hyperparameters["activations"]:
#             # Initialize W&B for this run
#             wandb.init(project="mlp-hyperparameter-tuning", name=f"lr={lr}_layers={hidden_layers}_activation={activation}", reinit=True)
            
#             # Initialize MLP model with hyperparameters
#             mlp = MLPClassifier(input_size=X_train.shape[1], hidden_layers=hidden_layers, output_size=num_classes, lr=lr, epochs=100, activation=activation)
            
#             # Train the model
#             mlp.fit(X_train, y_train_one_hot, X_val=X_test, y_val=y_test_one_hot)

#             # Make predictions on test data
#             y_pred = mlp.predict(X_test)

#             # Compute precision, recall, and F1-score
#             y_test_labels = y_test_shifted
#             precision = precision_score(y_test_labels, y_pred, average="weighted")
#             recall = recall_score(y_test_labels, y_pred, average="weighted")
#             f1 = f1_score(y_test_labels, y_pred, average="weighted")

#             # Log metrics to W&B
#             wandb.log({
#                 "learning_rate": lr,
#                 "hidden_layers": hidden_layers,
#                 "activation": activation,
#                 "precision": precision,
#                 "recall": recall,
#                 "f1_score": f1,
#                 "train_loss": mlp.train_loss_history[-1],
#                 "val_loss": mlp.val_loss_history[-1],
#                 "train_accuracy": mlp.train_acc_history[-1],
#                 "val_accuracy": mlp.val_acc_history[-1],
#                 "train_loss_history": mlp.train_loss_history,
#                 "val_loss_history": mlp.val_loss_history,
#                 "train_accuracy_history": mlp.train_acc_history,
#                 "val_accuracy_history": mlp.val_acc_history
#             })

#             # Track the best model based on F1-score
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_hyperparams = {"learning_rate": lr, "hidden_layers": hidden_layers, "activation": activation}

# # Report the best hyperparameters and their corresponding metrics
# print(f"Best Hyperparameters: {best_hyperparams}")
# print(f"Best F1-Score: {best_f1:.4f}")

# Best Hyperparameters: {'learning_rate': 0.1, 'hidden_layers': [128, 64, 32], 'activation': 'tanh'}
# Best F1-Score: 0.1931


# End the W&B run
wandb.finish()




# import numpy as np
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# import wandb

# # Initialize W&B for best model assessment
# wandb.init(project="mlp-hyperparameter-tuning", name="Best_Model_Evaluation")

# # Best hyperparameters from tuning
# best_hyperparams = {'learning_rate': 0.1, 'hidden_layers': [128, 64, 32], 'activation': 'tanh'}

# # Initialize MLP model with best hyperparameters
# best_mlp = MLPClassifier(
#     input_size=X_train.shape[1], 
#     hidden_layers=best_hyperparams['hidden_layers'], 
#     output_size=num_classes, 
#     lr=best_hyperparams['learning_rate'], 
#     epochs=100, 
#     activation=best_hyperparams['activation']
# )

# # Train the model using the best hyperparameters
# best_mlp.fit(X_train, y_train_one_hot, X_val=X_test, y_val=y_test_one_hot)

# # Predict on the test set
# y_pred = best_mlp.predict(X_test)

# # Convert one-hot encoded labels back to original form for comparison
# y_test_labels = y_test_shifted

# # Calculate metrics
# accuracy = accuracy_score(y_test_labels, y_pred)
# precision = precision_score(y_test_labels, y_pred, average="weighted")
# recall = recall_score(y_test_labels, y_pred, average="weighted")
# f1 = f1_score(y_test_labels, y_pred, average="weighted")

# # Log the metrics to W&B
# wandb.log({
#     "accuracy": accuracy,
#     "precision": precision,
#     "recall": recall,
#     "f1_score": f1
# })

# # Output the results
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-Score: {f1:.4f}")

# # End W&B run
# wandb.finish()


# Initialize W&B for activation function comparison
wandb.init(project="mlp-convergence-analysis", name="Effect_of_Activation_Functions")

# Activation functions to test
activation_functions = ["sigmoid", "tanh", "relu", "linear"]

# Dictionary to store losses for plotting
loss_history = {}

for activation in activation_functions:
    # Initialize the model with varying activation functions
    mlp = MLPClassifier(
        input_size=X_train.shape[1], 
        hidden_layers=[128, 64, 32], 
        output_size=num_classes, 
        lr=0.1,  # Constant learning rate
        epochs=100, 
        activation=activation,  # Varying activation function
        batch_size=32  # Constant batch size
    )
    
    # Train the model and log loss history
    mlp.fit(X_train, y_train_one_hot, X_val=X_test, y_val=y_test_one_hot)
    
    # Store loss history for each activation function
    loss_history[activation] = mlp.train_loss_history

    # Log losses to W&B
    for epoch, loss in enumerate(mlp.train_loss_history):
        wandb.log({"activation": activation, "epoch": epoch, "loss": loss})

wandb.finish()


# Initialize W&B for learning rate comparison
wandb.init(project="mlp-convergence-analysis", name="Effect_of_Learning_Rate")

# Learning rates to test
learning_rates = [0.001, 0.01, 0.1, 0.5]

# Dictionary to store losses for plotting
loss_history = {}

for lr in learning_rates:
    # Initialize the model with varying learning rates
    mlp = MLPClassifier(
        input_size=X_train.shape[1], 
        hidden_layers=[128, 64, 32], 
        output_size=num_classes, 
        lr=lr,  # Varying learning rate
        epochs=100, 
        activation="tanh",  # Constant activation function
        batch_size=32  # Constant batch size
    )
    
    # Train the model and log loss history
    mlp.fit(X_train, y_train_one_hot, X_val=X_test, y_val=y_test_one_hot)
    
    # Store loss history for each learning rate
    loss_history[lr] = mlp.train_loss_history

    # Log losses to W&B
    for epoch, loss in enumerate(mlp.train_loss_history):
        wandb.log({"learning_rate": lr, "epoch": epoch, "loss": loss})

wandb.finish()


# Initialize W&B for batch size comparison
wandb.init(project="mlp-convergence-analysis", name="Effect_of_Batch_Size")

# Batch sizes to test
batch_sizes = [16, 32, 64, 128]

# Dictionary to store losses for plotting
loss_history = {}

for batch_size in batch_sizes:
    # Initialize the model with varying batch sizes
    mlp = MLPClassifier(
        input_size=X_train.shape[1], 
        hidden_layers=[128, 64, 32], 
        output_size=num_classes, 
        lr=0.1,  # Constant learning rate
        epochs=100, 
        activation="tanh",  # Constant activation function
        batch_size=batch_size  # Varying batch size
    )
    
    # Train the model and log loss history
    mlp.fit(X_train, y_train_one_hot, X_val=X_test, y_val=y_test_one_hot)
    
    # Store loss history for each batch size
    loss_history[batch_size] = mlp.train_loss_history

    # Log losses to W&B
    for epoch, loss in enumerate(mlp.train_loss_history):
        wandb.log({"batch_size": batch_size, "epoch": epoch, "loss": loss})

wandb.finish()



########################################################################################################################################################






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from models.MLP.multilableMLP import MultiLabelMLPClassifier


def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    test_size = int(len(indices) * test_size)
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Custom Label Encoding function
def custom_label_encoder(column):
    unique_vals = sorted(list(set(column)))
    val_to_num = {val: num for num, val in enumerate(unique_vals)}
    encoded_column = column.map(val_to_num)
    return encoded_column, val_to_num

# Custom MultiLabel Binarizer for multi-label encoding
def custom_multilabel_binarizer(label_lists, unique_labels):
    label_set = set(unique_labels)
    binarized_output = np.zeros((len(label_lists), len(label_set)), dtype=int)
    
    for i, label_list in enumerate(label_lists):
        for label in label_list:
            if label in label_set:
                binarized_output[i, unique_labels.index(label)] = 1
    return binarized_output

# Custom Standard Scaler for feature scaling
def custom_standard_scaler(data):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    scaled_data = (data - means) / stds
    return scaled_data, means, stds


# Load the dataset
df = pd.read_csv('./data/external/advertisement.csv')

# Step 1: Split the 'labels' column into individual labels
df['labels'] = df['labels'].apply(lambda x: x.split())

# Step 2: Label encoding of categorical features
categorical_features = ['gender', 'education', 'married', 'city', 'occupation', 'most bought item']
label_encoders = {}

# Apply custom label encoding to each categorical feature
for feature in categorical_features:
    df[feature], encoder = custom_label_encoder(df[feature])
    label_encoders[feature] = encoder  # Store the encoder for each feature if you need it later

# Step 3: Label encoding for multi-label classification (using custom MultiLabelBinarizer)
unique_labels = sorted(list(set([label for sublist in df['labels'] for label in sublist])))
labels_encoded = custom_multilabel_binarizer(df['labels'], unique_labels)

# Step 4: Scaling continuous features using custom standard scaler
continuous_features = ['age', 'income', 'purchase_amount']
continuous_scaled, means, stds = custom_standard_scaler(df[continuous_features].values)

# Step 5: Combine preprocessed features
X = pd.DataFrame(continuous_scaled, columns=continuous_features)
X = pd.concat([X, df[categorical_features].reset_index(drop=True)], axis=1)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = custom_train_test_split(X.to_numpy(), labels_encoded, test_size=0.2, random_state=42)

# Plot distribution of features
df.hist(bins=30, figsize=(15, 10), layout=(5, 4))
plt.suptitle('Feature Distributions')
plt.tight_layout()
# plt.show()

# Convert y_train and y_test to a binary matrix representation for multi-label classification
# This is done with the custom_multilabel_binarizer above.

# Ensure X_train is a NumPy array
X_train_np = X_train if isinstance(X_train, np.ndarray) else X_train.to_numpy()
y_train_np = y_train  # Assuming y_train is already in binary multi-label format

# Fit the multi-label MLP model
mlp_multi = UnifiedMLP(input_size=X_train_np.shape[1], hidden_layers=[64, 32], output_size=8, lr=0.01, epochs=50, task="multilabel", activation="sigmoid")

# Train the model
mlp_multi.fit(X_train_np, y_train_np)

# Evaluate the model
# evaluation = mlp_multi.evaluate(X_test, y_test)

# Print the evaluation metrics
# print("Evaluation:", evaluation)





############################################################################################################################################################



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.MLP.MLP import MLPClassifier
# from models.MLP.multilableMLP import MultiLabelMLPClassifier
# from models.MLP.MLPreg import MLPRegressor
# from models.MLP.MLP import MLPClassifier
# from models.MLP.MLPreg import MLPBinaryClassifier

import pandas as pd



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import wandb
# Assuming MLPRegressor is already defined as shown in the previous code block



# Load dataset from CSV file
try:
    data = pd.read_csv("./data/external/HousingData.csv")
except FileNotFoundError:
    print("The file  was not found. Please check the path and filename.")
    exit()

# Check for missing values in the dataset
print("Missing values per column:")
print(data.isnull().sum())

# Option 1: Remove rows with missing values
data_cleaned = data.dropna()

# Option 2: Impute missing values with column means (use this if you prefer to keep all data)
# data_cleaned = data.fillna(data.mean())

# Separate features (X) and target (y)
X = data_cleaned.iloc[:, :-1].values  # Features (all columns except the last)
y = data_cleaned.iloc[:, -1].values   # Target (the last column)

# 1. Dataset Description (after cleaning)
print(f"Mean of each feature: {np.mean(X, axis=0)}")
print(f"Std Dev of each feature: {np.std(X, axis=0)}")
print(f"Min of each feature: {np.min(X, axis=0)}")
print(f"Max of each feature: {np.max(X, axis=0)}")

# 2. Label Distribution Graph
plt.hist(y, bins=30, edgecolor='black')
plt.title('Distribution of MEDV (Housing Prices)')
plt.xlabel('MEDV')
plt.ylabel('Frequency')
plt.show()

# 3. Partitioning the dataset
X_train, X_temp, y_train, y_temp = custom_train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = custom_train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 4. Normalizing and Standardizing the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# # 5. MLP Training with regression on the housing dataset
# mlp = MLPRegressor(
#     input_size=X_train.shape[1], 
#     hidden_layers=[64, 32], 
#     output_size=1, 
#     lr=0.0001,  # Lower learning rate
#     epochs=100, 
#     batch_size=32, 
#     activation="relu",  # Keep ReLU for hidden layers
#     optimizer="sgd"
# )

# mlp.fit(X_train, y_train)

# # Evaluate performance on validation and test sets
# val_predictions = mlp.predict(X_val)
# test_predictions = mlp.predict(X_test)

# val_loss = np.mean((val_predictions - y_val) ** 2)
# test_loss = np.mean((test_predictions - y_test) ** 2)

# print(f"Validation MSE: {val_loss}")
# print(f"Test MSE: {test_loss}")

# # Optionally plot predicted vs actual values
# plt.scatter(y_test, test_predictions, alpha=0.5)
# plt.title('Actual vs Predicted Housing Prices')
# plt.xlabel('Actual Prices (MEDV)')
# plt.ylabel('Predicted Prices')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line for perfect predictions
# plt.show()


# import wandb

# # Initialize W&B
# wandb.init(project="mlp-regressor-housing", name="MLP-Regressor-Run")

# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np

# # 1. Compute Metrics after predictions
# val_predictions = mlp.predict(X_val)
# test_predictions = mlp.predict(X_test)

# # Calculate Mean Squared Error (MSE)
# val_mse = mean_squared_error(y_val, val_predictions)
# test_mse = mean_squared_error(y_test, test_predictions)

# # Calculate Root Mean Squared Error (RMSE)
# val_rmse = np.sqrt(val_mse)
# test_rmse = np.sqrt(test_mse)

# # Calculate R-squared score (R^2)
# val_r2 = r2_score(y_val, val_predictions)
# test_r2 = r2_score(y_test, test_predictions)

# # 2. Log these metrics to W&B
# wandb.log({
#     "Validation MSE": val_mse,
#     "Validation RMSE": val_rmse,
#     "Validation R-squared": val_r2,
#     "Test MSE": test_mse,
#     "Test RMSE": test_rmse,
#     "Test R-squared": test_r2
# })

# # Optionally print metrics for local visibility
# print(f"Validation MSE: {val_mse}")
# print(f"Validation RMSE: {val_rmse}")
# print(f"Validation R-squared: {val_r2}")
# print(f"Test MSE: {test_mse}")
# print(f"Test RMSE: {test_rmse}")
# print(f"Test R-squared: {test_r2}")

# wandb.finish()


# sweep_config = {
#     'method': 'grid',  # or 'random'
#     'metric': {
#       'name': 'Validation MSE',
#       'goal': 'minimize'   
#     },
#     'parameters': {
#         'lr': {
#             'values': [0.001, 0.0001, 0.00001]  # Learning rates to test
#         },
#         'activation': {
#             'values': ['relu', 'tanh', 'sigmoid']  # Different activation functions to test
#         },
#         'optimizer': {
#             'values': ['sgd', 'adam']  # Different optimizers to test
#         },
#         'hidden_layers': {
#             'values': [[64, 32], [128, 64], [32]]  # Different hidden layer sizes to test
#         }
#     }
# }

# def train_model_with_sweep():
#     # Initialize a W&B run
#     run = wandb.init()

#     # Fetch sweep parameters
#     config = wandb.config

#     # Instantiate and train MLP Regressor using the current hyperparameters
#     mlp = MLPRegressor(
#         input_size=X_train.shape[1], 
#         hidden_layers=config.hidden_layers, 
#         output_size=1, 
#         lr=config.lr, 
#         epochs=100, 
#         batch_size=32, 
#         activation=config.activation, 
#         optimizer=config.optimizer
#     )

#     mlp.fit(X_train, y_train)

#     # Evaluate model
#     val_predictions = mlp.predict(X_val)
#     val_mse = mean_squared_error(y_val, val_predictions)

#     # Log the loss (MSE) for the current set of hyperparameters
#     wandb.log({
#         "Validation MSE": val_mse
#     })

#     # Finalize the run
#     wandb.finish()

# # Initialize the sweep
# sweep_id = wandb.sweep(sweep_config, project="mlp-hyperparameter-tuning")

# # Run the sweep
# wandb.agent(sweep_id, function=train_model_with_sweep, count=20)

# import pandas as pd

# # Create a dataframe to store the results
# results = []

# for run in wandb.Api().sweep(sweep_id).runs:
#     results.append({
#         "learning_rate": run.config['lr'],
#         "activation": run.config['activation'],
#         "optimizer": run.config['optimizer'],
#         "hidden_layers": run.config['hidden_layers'],
#         "Validation MSE": run.summary['Validation MSE'],
#     })

# # Convert to DataFrame
# df_results = pd.DataFrame(results)

# # Display the table
# print(df_results)


# # Find the best configuration based on Validation MSE
# best_model = df_results.loc[df_results['Validation MSE'].idxmin()]

# print("Best Model Hyperparameters:")
# print(best_model)


# import matplotlib.pyplot as plt

# # Train with BCE Loss
# bce_model = MLPBinaryClassifier(input_size=X_train.shape[1], hidden_layers=[], lr=0.001, epochs=100, loss_fn="bce")
# bce_model.fit(X_train, y_train)

# # Train with MSE Loss
# mse_model = MLPBinaryClassifier(input_size=X_train.shape[1], hidden_layers=[], lr=0.001, epochs=100, loss_fn="mse")
# mse_model.fit(X_train, y_train)

# # Plot Loss vs Epochs for BCE Loss
# plt.plot(bce_model.losses, label="BCE Loss")
# plt.title("Loss vs Epochs (BCE)")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.show()

# # Plot Loss vs Epochs for MSE Loss
# plt.plot(mse_model.losses, label="MSE Loss")
# plt.title("Loss vs Epochs (MSE)")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import wandb

# Load the dataset
data = pd.read_csv("./data/external/diabetes.csv")
X = data.iloc[:, :-1].values  # Features (all columns except the last)
y = data.iloc[:, -1].values   # Target (last column)


plt.hist(y, bins=30, edgecolor='black')
plt.title('Distribution of MEDV (Housing Prices)')
plt.xlabel('MEDV')
plt.ylabel('Frequency')
# plt.show()

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = custom_train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Model training parameters
epochs = 100
lr = 0.01
batch_size = 32

# Initialize W&B for tracking
wandb.init(project="binary_classification", name="MSE_vs_BCE")

# 1. Model using MSE loss
model_mse = MLPClassifier(input_size=X_train.shape[1], hidden_layers=[], output_size=1, lr=lr, epochs=epochs, batch_size=batch_size, activation='sigmoid')
model_mse.fit(X_train, y_train)

# 2. Model using BCE loss
model_bce = MLPClassifier(input_size=X_train.shape[1], hidden_layers=[], output_size=1, lr=lr, epochs=epochs, batch_size=batch_size, activation='sigmoid')
model_bce.fit(X_train, y_train)

# Plotting loss vs epochs
epochs_range = np.arange(1, epochs + 1)

# MSE loss plot
plt.plot(epochs_range, model_mse.loss_history, label='MSE Loss')
plt.title('Loss vs Epochs (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# BCE loss plot
plt.plot(epochs_range, model_bce.loss_history, label='BCE Loss', color='red')
plt.title('Loss vs Epochs (BCE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# W&B Logging
wandb.finish()






############################################################################################################################################################

# import pandas as pd
# import numpy as np
# import os
# import sys
# from sklearn.preprocessing import LabelEncoder

# # from sklearn.neighbors import 
# from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# from models.knn.knn import KNNClassifier as KNeighborsClassifier
# # Assuming AutoEncoder is implemented as per previous discussion
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from models.MLP.autoencoder import AutoEncoder
# from models.MLP.MLP import MLPClassifier
# from models.MLP.combined import UnifiedMLP


# def custom_train_test_split(X, y, test_size=0.2, random_state=42):
#     np.random.seed(random_state)
#     indices = np.arange(X.shape[0])
#     np.random.shuffle(indices)
    
#     test_size = int(len(indices) * test_size)
#     train_indices = indices[:-test_size]
#     test_indices = indices[-test_size:]
    
#     X_train, X_test = X[train_indices], X[test_indices]
#     y_train, y_test = y[train_indices], y[test_indices]
    
#     return X_train, X_test, y_train, y_test


# # Load the dataset
# def load_data(csv_file):
#     data = pd.read_csv(csv_file)
    
#     # Dropping unnecessary columns
#     data = data.drop(['track_id', 'artists', 'album_name', 'track_name', 'key', 'mode', 'liveness', 'valence'], axis=1)
#     data = data.dropna(axis=0)  # Removing rows with missing values
    
#     # Standardize feature columns
#     feature_columns = data.columns[:-1]
#     data[feature_columns] = (data[feature_columns] - data[feature_columns].mean()) / data[feature_columns].std()
    
#     X = data.iloc[:, :-1].values  # Features
#     y = data.iloc[:, -1].values   # Target variable (used for KNN classification)
#     return X, y

# # # Main code
# csv_file = './data/external/spotify.csv'  # Path to the dataset
# X, y = load_data(csv_file)

# # Split data into training and validation sets
# X_train, X_val, y_train, y_val = custom_train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the autoencoder
# autoencoder = AutoEncoder(input_size=12, hidden_layers=[10, 9], latent_dim=9, lr=0.01, epochs=50, batch_size=32, activation="sigmoid")

# # Train the autoencoder on X_train
# autoencoder.fit(X_train)

# # Get latent representation of the training and validation sets
# X_train_reduced = autoencoder.get_latent(X_train)
# X_val_reduced = autoencoder.get_latent(X_val)

# # Apply KNN on the reduced dataset (latent space)
# knn = KNeighborsClassifier(n_neighbors=28)

# knn.fit(X_train_reduced, y_train)

# # Predict on validation set
# y_val_pred = knn.predict(X_val_reduced)

# # Calculate evaluation metrics
# f1 = f1_score(y_val, y_val_pred, average='macro')
# accuracy = accuracy_score(y_val, y_val_pred)
# precision = precision_score(y_val, y_val_pred, average='macro')
# recall = recall_score(y_val, y_val_pred, average='macro')

# # Output the results
# print(f"F1 Score: {f1:.4f}")
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")



# csv_file = './data/external/spotify.csv'
# X, y = load_data(csv_file)

# # Split data into training and validation sets
# X_train, X_val, y_train, y_val = custom_train_test_split(X, y, test_size=0.2, random_state=42)
# # Ensure the target labels are integers


# # Convert string labels to integer labels
# label_encoder = LabelEncoder()
# y_train = label_encoder.fit_transform(y_train)
# y_val = label_encoder.transform(y_val)

# # One-hot encode the target variable for MLP
# num_classes = len(np.unique(y_train))
# y_train_one_hot = np.eye(num_classes)[y_train]
# y_val_one_hot = np.eye(num_classes)[y_val]


# # One-hot encode the target variable for MLP
# num_classes = len(np.unique(y_train))
# y_train_one_hot = np.eye(num_classes)[y_train]
# y_val_one_hot = np.eye(num_classes)[y_val]



# # Initialize and train the MLP classifier
# # mlp_classifier = MLPClassifier(input_size=X_train.shape[1], hidden_layers=[12, 10], output_size=num_classes, 
#                             #    lr=0.01, epochs=100, batch_size=32, activation="relu", optimizer="sgd")

# mlp_classifier = UnifiedMLP(input_size=X_train.shape[1], hidden_layers=[12, 10], output_size=num_classes, task="classification",
#                                lr=0.01, epochs=100, batch_size=32, activation="relu", optimizer="sgd")



# mlp_classifier.fit(X_train, y_train_one_hot)

# # Predict on validation set
# y_val_pred = mlp_classifier.predict(X_val)

# # Calculate metrics
# f1 = f1_score(y_val, y_val_pred, average='macro')
# accuracy = accuracy_score(y_val, y_val_pred)
# precision = precision_score(y_val, y_val_pred, average='macro')
# recall = recall_score(y_val, y_val_pred, average='macro')

# print(f"MLP Classifier Results:")
# print(f"F1 Score: {f1}")
# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")



# # MLP Classifier Results:
# # F1 Score: 0.19435797057333884
# # Accuracy: 0.2693421052631579
# # Precision: 0.22534819388827448
# # Recall: 0.2656405565912657


import pandas as pd
import numpy as np
import os
import sys
import pandas as pd
import numpy as np
import os
import sys
# from sklearn.preprocessing import LabelEncoder

# from sklearn.neighbors import 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from models.knn.knn import KNNClassifier as KNeighborsClassifier
# Assuming AutoEncoder is implemented as per previous discussion
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.AutoEncoders import AutoEncoders
# from models.MLP.MLP import MLPClassifier
from models.MLP.MLP import UnifiedMLP

# Custom function to split the dataset into training and testing sets
def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    test_size = int(len(indices) * test_size)
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Custom function for label encoding
def custom_label_encoder(y):
    classes = np.unique(y)
    class_map = {label: idx for idx, label in enumerate(classes)}
    y_encoded = np.array([class_map[label] for label in y])
    return y_encoded, class_map

# Custom metric functions
def custom_accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def custom_precision_score(y_true, y_pred):
    unique_labels = np.unique(y_true)
    precision_list = []
    
    for label in unique_labels:
        true_positives = np.sum((y_pred == label) & (y_true == label))
        predicted_positives = np.sum(y_pred == label)
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        precision_list.append(precision)
    
    return np.mean(precision_list)

def custom_recall_score(y_true, y_pred):
    unique_labels = np.unique(y_true)
    recall_list = []
    
    for label in unique_labels:
        true_positives = np.sum((y_pred == label) & (y_true == label))
        actual_positives = np.sum(y_true == label)
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        recall_list.append(recall)
    
    return np.mean(recall_list)

def custom_f1_score(y_true, y_pred):
    precision = custom_precision_score(y_true, y_pred)
    recall = custom_recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Load the dataset
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    
    # Dropping unnecessary columns
    data = data.drop(['track_id', 'artists', 'album_name', 'track_name', 'key', 'mode', 'liveness', 'valence',], axis=1)
    data = data.dropna(axis=0)  # Removing rows with missing values
    
    # Standardize feature columns
    feature_columns = data.columns[:-1]
    data[feature_columns] = (data[feature_columns] - data[feature_columns].mean()) / data[feature_columns].std()
    
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Target variable (used for KNN classification)
    return X, y

# Main code
csv_file = './data/external/spotify.csv'  # Path to the dataset
X, y = load_data(csv_file)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = custom_train_test_split(X, y, test_size=0.2, random_state=42)

# Convert string labels to integer labels using custom label encoder
y_train, class_map = custom_label_encoder(y_train)
y_val = np.array([class_map[label] for label in y_val])

# One-hot encode the target variable for MLP
num_classes = len(np.unique(y_train))
y_train_one_hot = np.eye(num_classes)[y_train]
y_val_one_hot = np.eye(num_classes)[y_val]

# Initialize and train the autoencoder
autoencoder = AutoEncoders(input_size=12, hidden_layers=[10, 9], latent_dim=9, lr=0.01, epochs=50, batch_size=32, activation="sigmoid")

# Train the autoencoder on X_train
autoencoder.fit(X_train)

# Get latent representation of the training and validation sets
X_train_reduced = autoencoder.get_latent(X_train)
X_val_reduced = autoencoder.get_latent(X_val)

# Apply KNN on the reduced dataset (latent space)
knn = KNeighborsClassifier(n_neighbors=28)
knn.fit(X_train_reduced, y_train)

# Predict on validation set
y_val_pred = knn.predict(X_val_reduced)

# Calculate evaluation metrics using custom functions
f1 = custom_f1_score(y_val, y_val_pred)
accuracy = custom_accuracy_score(y_val, y_val_pred)
precision = custom_precision_score(y_val, y_val_pred)
recall = custom_recall_score(y_val, y_val_pred)

# Output the results
print(f"KNN Classifier Results:")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Initialize and train the MLP classifier
mlp_classifier = UnifiedMLP(input_size=X_train.shape[1], hidden_layers=[12, 10], output_size=num_classes, task="classification",
                            lr=0.01, epochs=100, batch_size=32, activation="relu", optimizer="sgd")

mlp_classifier.fit(X_train, y_train_one_hot)

# Predict on validation set
y_val_pred = mlp_classifier.predict(X_val)

# Calculate metrics using custom functions
f1 = custom_f1_score(y_val, y_val_pred)
accuracy = custom_accuracy_score(y_val, y_val_pred)
precision = custom_precision_score(y_val, y_val_pred)
recall = custom_recall_score(y_val, y_val_pred)

print(f"MLP Classifier Results:")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
