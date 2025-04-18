import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsClassifier
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.knn import knn 

 


def load_data(csv_file):
    data = pd.read_csv(csv_file)
    
    data = data.drop(['track_id', 'artists', 'album_name', 'track_name','key', 'mode', 'liveness', 'valence'], axis=1)
    data = data.dropna(axis=0)
    
    feature_columns = data.columns[:-1]
    data[feature_columns] = (data[feature_columns] - data[feature_columns].mean()) / data[feature_columns].std()
    
    X = data.iloc[:, :-1].values  
    y = data.iloc[:, -1].values   
    return X, y

def split_data(X, y, train_ratio=0.8, validate_ratio=0.10):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    train_size = int(train_ratio * X.shape[0])
    validate_size = int(validate_ratio * X.shape[0])

    X_train, y_train = X[:train_size], y[:train_size]
    X_validate, y_validate = X[train_size:train_size + validate_size], y[train_size:train_size + validate_size]
    X_test, y_test = X[train_size + validate_size:], y[train_size + validate_size:]

    return X_train, X_validate, X_test, y_train, y_validate, y_test

def run_knn(csv_file, k=3, distance_metric='euclidean',run_calc='none'):
    X, y = load_data(csv_file)

    X_train, X_validate, X_test, y_train, y_validate, y_test = split_data(X, y)

    model = knn.KNN(k=k, distance_metric=distance_metric)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_validate)
    # print(y_train)
    

    
    metrics = knn.Metrics(label_mapping=model.label_mapping)
    accuracy = metrics.accuracy(y_validate, y_pred)
    
    
    if(run_calc=='test'):

        precision_macro = metrics.precision(y_validate, y_pred, average='macro')
        recall_macro = metrics.recall(y_validate, y_pred, average='macro')
        f1_macro = metrics.f1_score(y_validate, y_pred, average='macro')
        
        precision_micro = metrics.precision(y_validate, y_pred, average='micro')
        recall_micro = metrics.recall(y_validate, y_pred, average='micro')
        f1_micro = metrics.f1_score(y_validate, y_pred, average='micro')

        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
        print(f"Validation Precision (Macro): {precision_macro * 100:.2f}%")
        print(f"Validation Recall (Macro): {recall_macro * 100:.2f}%")
        print(f"Validation F1 Score (Macro): {f1_macro * 100:.2f}%")
        print(f"Validation Precision (Micro): {precision_micro * 100:.2f}%")
        print(f"Validation Recall (Micro): {recall_micro * 100:.2f}%")
        print(f"Validation F1 Score (Micro): {f1_micro * 100:.2f}%")

    return accuracy
    

def plot_k_vs_accuracy(csv_file, distance_metric, k_values):
    accuracies = []

    for k in k_values:
        accuracy = run_knn(csv_file, k, distance_metric)
        accuracies.append(accuracy*100)
        print(f"k={k}, accuracy={accuracy:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title(f'k vs Accuracy for {distance_metric.capitalize()} Distance')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig(f'k_vs_accuracy_{distance_metric}.png')
    plt.close()


csv_file = './data/external/spotify.csv'  
k_values = range(28, 32)
plot_k_vs_accuracy(csv_file, 'manhattan', k_values)
# ac1=run_knn(csv_file, k=10, distance_metric='manhattan')
# print(ac1)
# 

results = []

for k in range(1, 50):
    for distance_metric in ['cosine', 'euclidean', 'manhattan']:
        accuracy = run_knn(csv_file, k, distance_metric)
        results.append((k, distance_metric, accuracy))

top_10_results = sorted(results, key=lambda x: x[2], reverse=True)[:10]

print("Top 10 k and distance metric pairs by accuracy:")
for rank, (k, distance_metric, accuracy) in enumerate(top_10_results, start=1):
    print(f"Rank {rank}: k={k}, distance_metric={distance_metric}, accuracy={accuracy:.4f}")



def plot_inference_time(csv_file, k_values, distance_metric):
    custom_knn_times = []
    sklearn_knn_times = []

    X, y = load_data(csv_file)
    X_train, X_validate, X_test, y_train, y_validate, y_test = split_data(X, y)
    

    for k in k_values:
        custom_knn = knn.KNN(k=k, distance_metric=distance_metric)
        custom_knn.fit(X_train, y_train)
        
        start_time = time.time()
        custom_knn.predict(X_test)
        end_time = time.time()
        custom_knn_times.append(end_time - start_time)
        
        sklearn_knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
        sklearn_knn.fit(X_train, y_train)
        
        start_time = time.time()
        sklearn_knn.predict(X_test)
        end_time = time.time()
        sklearn_knn_times.append(end_time - start_time)

        print(f"k={k} | Custom KNN Time: {custom_knn_times[-1]:.4f}s | Sklearn KNN Time: {sklearn_knn_times[-1]:.4f}s")
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, custom_knn_times, marker='o', linestyle='-', color='blue', label='Custom KNN')
    plt.plot(k_values, sklearn_knn_times, marker='o', linestyle='-', color='red', label='Sklearn KNN')
    plt.title(f'Inference Time Comparison for {distance_metric.capitalize()} Distance')
    plt.xlabel('k')
    plt.ylabel('Inference Time (seconds)')
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'inference time my_knn vs sklearn_knn.png')
    plt.close()



distance_metric = 'euclidean'  
k_values = range(1, 11)  

plot_inference_time(csv_file, k_values, distance_metric)

# ac1=run_knn(csv_file, k=10, distance_metric='manhattan')
# print(ac1)


def plot_inference_time_vs_dataset_size(csv_file, k, distance_metric):
    custom_knn_times = []
    sklearn_knn_times = []
    dataset_sizes = []

    X, y = load_data(csv_file)
    X_train_full, X_validate, X_test, y_train_full, y_validate, y_test = split_data(X, y)

    percentages = np.arange(0.1, 1.1, 0.1)
    
    for p in percentages:
        size = int(len(X_train_full) * p)
        X_train = X_train_full[:size]
        y_train = y_train_full[:size]
        dataset_sizes.append(size)

        custom_knn = knn.KNN(k=k, distance_metric=distance_metric)
        custom_knn.fit(X_train, y_train)
        
        start_time = time.time()
        custom_knn.predict(X_test)
        end_time = time.time()
        custom_knn_times.append(end_time - start_time)
        
        sklearn_knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
        sklearn_knn.fit(X_train, y_train)
        
        start_time = time.time()
        sklearn_knn.predict(X_test)
        end_time = time.time()
        sklearn_knn_times.append(end_time - start_time)

        print(f"Dataset Size: {size} | Custom KNN Time: {custom_knn_times[-1]:.4f}s | Sklearn KNN Time: {sklearn_knn_times[-1]:.4f}s")
    
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, custom_knn_times, marker='o', linestyle='-', color='blue', label='Custom KNN')
    plt.plot(dataset_sizes, sklearn_knn_times, marker='o', linestyle='-', color='red', label='Sklearn KNN')
    plt.title(f'Inference Time vs Dataset Size for {distance_metric.capitalize()} Distance (k={k})')
    plt.xlabel('Train Dataset Size')
    plt.ylabel('Inference Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('inference_time_vs_dataset_size.png')
    plt.close()


k = 5  
distance_metric = 'manhattan'  

plot_inference_time_vs_dataset_size(csv_file, k, distance_metric)


csv_test = './data/external/spotify-2/test.csv'
csv_train = './data/external/spotify-2/train.csv'
csv_validate = './data/external/spotify-2/validate.csv'

X2_test, y2_test=load_data(csv_test)
X2_train, y2_train=load_data(csv_train)
X2_validate, y2_validate=load_data(csv_validate)
model2 = knn.KNN(k=3)
model2.fit(X2_train, y2_train)

unique_labels = np.unique(y2_test)
for label in unique_labels:
    if label not in model2.label_mapping:
        model2.label_mapping[label] = len(model2.label_mapping)

y_pred = model2.predict(X2_test)
metrics = knn.Metrics(label_mapping=model2.label_mapping)
accuracy = metrics.accuracy(y2_test, y_pred)

precision_macro = metrics.precision(y2_test, y_pred, average='macro')
recall_macro = metrics.recall(y2_test, y_pred, average='macro')
f1_macro = metrics.f1_score(y2_test, y_pred, average='macro')
        
precision_micro = metrics.precision(y2_test, y_pred, average='micro')
recall_micro = metrics.recall(y2_test, y_pred, average='micro')
f1_micro = metrics.f1_score(y2_test, y_pred, average='micro')

print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Precision (Macro): {precision_macro * 100:.2f}%")
print(f"Validation Recall (Macro): {recall_macro * 100:.2f}%")
print(f"Validation F1 Score (Macro): {f1_macro * 100:.2f}%")
print(f"Validation Precision (Micro): {precision_micro * 100:.2f}%")
print(f"Validation Recall (Micro): {recall_micro * 100:.2f}%")
print(f"Validation F1 Score (Micro): {f1_micro * 100:.2f}%")

# ac1=run_knn(csv_file, k=10, distance_metric='manhattan')
# print(ac1)




#KNN code ends here.................LINEAR REGRESSION CODE STARTS HERE>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import numpy as np
import sys
import pandas as pd
import os
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib.pyplot as plt
from models.linear_regression.linear_regression import LinearRegression
from models.linear_regression.linear_regression import PolynomialRegression

data = pd.read_csv('.\data\external\linreg.csv')
X = data.iloc[:, 0].values.reshape(-1, 1)  
y = data.iloc[:, 1].values
def shuffle_and_split(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    train_split = int(len(X) * train_ratio)
    val_split = train_split + int(len(X) * val_ratio)
    
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    return (X[train_indices], y[train_indices],
            X[val_indices], y[val_indices],
            X[test_indices], y[test_indices])

def normalize_features(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

X_train, y_train, X_val, y_val, X_test, y_test = shuffle_and_split(X, y)

X_train_norm, mean, std = normalize_features(X_train)
X_val_norm, _, _ = normalize_features(X_val, mean, std)
X_test_norm, _, _ = normalize_features(X_test, mean, std)

def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    return mse, r2

def calculate_metrics2(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    variance = np.var(y_pred)
    std_dev = np.std(y_pred)
    return mse, variance, std_dev

def experiment_with_learning_rates(learning_rates, X_train, y_train, X_val, y_val, X_test, y_test):
    best_lr = None
    best_val_score = -np.inf
    results = []

    for lr in learning_rates:
        model = LinearRegression(learning_rate=lr)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        train_mse, train_r2 = calculate_metrics(y_train, y_train_pred)
        val_mse, val_r2 = calculate_metrics(y_val, y_val_pred)
        test_mse, test_r2 = calculate_metrics(y_test, y_test_pred)

        results.append({
            "learning_rate": lr,
            "train_mse": train_mse,
            "train_r2": train_r2,
            "val_mse": val_mse,
            "val_r2": val_r2,
            "test_mse": test_mse,
            "test_r2": test_r2
        })

        if val_r2 > best_val_score:
            best_val_score = val_r2
            best_lr = lr

    return results, best_lr







X_train, y_train, X_val, y_val, X_test, y_test = shuffle_and_split(X, y)

X_train_norm, mean, std = normalize_features(X_train)
X_val_norm, _, _ = normalize_features(X_val, mean, std)
X_test_norm, _, _ = normalize_features(X_test, mean, std)
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5]
results, best_lr = experiment_with_learning_rates(learning_rates, X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test)

print("Best Learning Rate:", best_lr)
for result in results:
    print(f"Learning Rate: {result['learning_rate']}, Train MSE: {result['train_mse']:.4f}, Val MSE: {result['val_mse']:.4f}, Test MSE: {result['test_mse']:.4f}")







lambda_values = [0.001, 0.01, 0.1, 1, 10,0]
best_val_mse = float('inf')
best_lambda = None

for lambda_ in lambda_values:
    model = LinearRegression(lambda_=lambda_)
    model.fit(X_train_norm, y_train)
    y_val_pred = model.predict(X_val_norm)
    val_mse, _ = calculate_metrics(y_val, y_val_pred)
    
    print(f"Lambda: {lambda_}, Validation MSE: {val_mse:.4f}")
    
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_lambda = lambda_

print(f"\nBest lambda: {best_lambda}")

model = LinearRegression(lambda_=best_lambda)
model.fit(X_train_norm, y_train)

y_train_pred = model.predict(X_train_norm)
y_val_pred = model.predict(X_val_norm)
y_test_pred = model.predict(X_test_norm)

train_mse, train_r2 = calculate_metrics(y_train, y_train_pred)
val_mse, val_r2 = calculate_metrics(y_val, y_val_pred)
test_mse, test_r2 = calculate_metrics(y_test, y_test_pred)

print(f"Train set - MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
print(f"Validation set - MSE: {val_mse:.4f}, R2: {val_r2:.4f}")
print(f"Test set - MSE: {test_mse:.4f}, R2: {test_r2:.4f}")

plt.figure(figsize=(12, 8))
plt.scatter(X_train, y_train, color='blue', label='Train', alpha=0.7)
plt.scatter(X_val, y_val, color='green', label='Validation', alpha=0.7)
plt.scatter(X_test, y_test, color='red', label='Test', alpha=0.7)

X_sorted = np.sort(X, axis=0)
X_sorted_norm = (X_sorted - mean) / std
y_pred_sorted = model.predict(X_sorted_norm)

plt.plot(X_sorted, y_pred_sorted, color='black', label='Prediction')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression: Train, Validation, and Test Sets')
plt.legend()
plt.savefig(f'./assignments/1/figures/linear_reg_all.png')
plt.close()



train_mse, train_var, train_std = calculate_metrics2(y_train, y_train_pred)
val_mse, val_var, val_std = calculate_metrics2(y_val, y_val_pred)

print(f"Train set - MSE: {train_mse:.4f}, Variance: {train_var:.4f}, Std Dev: {train_std:.4f}")
print(f"Test set - MSE: {test_mse:.4f}, Variance: {val_var:.4f}, Std Dev: {val_std:.4f}")

β1 = model.coefficients[0] / std[0]
β0 = model.intercept - β1 * mean[0]

print(f"Fitted line: y = {β1:.4f}x + {β0:.4f}")

plt.figure(figsize=(12, 8))
plt.scatter(X_train, y_train, color='blue', label='Train', alpha=0.7)

X_line = np.array([X.min(), X.max()]).reshape(-1, 1)
y_line = β1 * X_line + β0

plt.plot(X_line, y_line, color='red', label='Fitted Line')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression: Training Points and Fitted Line')
plt.legend()
plt.savefig(f'./assignments/1/figures/linear_reg_train.png')
plt.close()



np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_val = X[indices[:split]], X[indices[split:]]
y_train, y_val = y[indices[:split]], y[indices[split:]]

max_degree = 10
results = []

for k in range(1, max_degree + 1):
    model = PolynomialRegression(degree=k)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_metrics = calculate_metrics2(y_train, y_train_pred)
    val_metrics = calculate_metrics2(y_val, y_val_pred)
    
    results.append((k, *train_metrics, *val_metrics))

print("Degree | Train MSE | Train StdDev | Train Var | Test MSE | Test StdDev | Test Var")
print("-" * 80)

best_k, best_val_mse = 0, float('inf')
for result in results:
    k, train_mse, train_std, train_var, val_mse, val_std, val_var = result
    print(f"{k:6d} | {train_mse:.4f} | {train_std:.4f} | {train_var:.4f} | {val_mse:.4f} | {val_std:.4f} | {val_var:.4f}")
    
    if val_mse < best_val_mse:
        best_k, best_val_mse = k, val_mse

print(f"\nBest k that minimizes error on test set: {best_k}")

best_model = PolynomialRegression(degree=best_k)
best_model.fit(X_train, y_train)
np.save('./assignments/1/best_model_params.npy', best_model.coefficients)





plt.figure(figsize=(12, 8))
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Train Data')
plt.scatter(X_test, y_test, color='red', alpha=0.5, label='Test Data')

X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_plot = best_model.predict(X_plot)
plt.plot(X_plot, y_plot, color='green', label=f'Best Fit (degree={best_k})')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression: Best Fit')
plt.legend()
plt.savefig(f'./assignments/1/figures/poly_regression_best_fit.png')
plt.close()





data = pd.read_csv("./data/external/regularisation.csv")

X = data.iloc[:, 0].values.reshape(-1, 1)  
y = data.iloc[:, 1].values

X_train, y_train, X_val, y_val, X_test, y_test = shuffle_and_split(X, y)
X_train = np.array(X_train, dtype=float)
X_val = np.array(X_val, dtype=float)
X_test = np.array(X_test, dtype=float)
X_train = np.array(X_train, dtype=float)
X_val = np.array(X_val, dtype=float)
X_test = np.array(X_test, dtype=float)

k = 5
model = PolynomialRegression(degree=k)
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))

plt.scatter(X_train, y_train, color='blue', label='Training Data')

plt.scatter(X_val, y_val, color='green', label='Validation Data')

plt.scatter(X_test, y_test, color='red', label='Test Data')

X_range = np.linspace(min(X_train), max(X_train), 100).reshape(-1, 1)
y_range_pred = model.predict(X_range)
plt.plot(X_range, y_range_pred, color='black', label=f'Polynomial Regression (degree={k})')

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Polynomial Regression with degree 5')
plt.savefig(f'./assignments/1/figures/ploy_reg_with_degree_5.png')
plt.close()
mse_val, variance_val,std_dev_val = calculate_metrics2(y_val, y_val_pred)
mse_test, variance_test,std_dev_test = calculate_metrics2(y_test, y_test_pred)

print(f'Validation Metrics:\nMSE: {mse_val}\nStandard Deviation: {std_dev_val}\nVariance: {variance_val}')
print(f'\nTest Metrics:\nMSE: {mse_test}\nStandard Deviation: {std_dev_test}\nVariance: {variance_test}')






def plot_polynomial_regression(X_train, y_train, X_test, y_test, degrees, alpha=0.01, learning_rate=0.001):
    

    for degree in degrees:
        plt.figure(figsize=(10, 6))
        
        model = PolynomialRegression(degree=degree)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        model_l2 = PolynomialRegression(degree=degree, regularization='L2', alpha=alpha)
        model_l2.fit(X_train, y_train)
        y_pred_l2 = model_l2.predict(X_test)
        
        model_l1 = PolynomialRegression(degree=degree, regularization='L1', alpha=alpha, learning_rate=learning_rate)
        model_l1.fit(X_train, y_train)
        y_pred_l1 = model_l1.predict(X_test)
        
        plt.scatter(X_test, y_test, color='blue', label='Test Data')
        
        X_range = np.linspace(min(X_train), max(X_train), 100).reshape(-1, 1)
        y_range_pred = model.predict(X_range)
        y_range_pred_l2 = model_l2.predict(X_range)
        y_range_pred_l1 = model_l1.predict(X_range)
        
        plt.plot(X_range, y_range_pred, color='black', label='No Regularization')
        plt.plot(X_range, y_range_pred_l2, color='red', linestyle='--', label='L2 Regularization')
        plt.plot(X_range, y_range_pred_l1, color='green', linestyle='--', label='L1 Regularization')
        
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'Polynomial Regression (Degree {degree})')
        
        plt.savefig(f'./assignments/1/figures/poly_regression_degree_{degree}.png')
        plt.close()

    for degree in degrees:
        model = PolynomialRegression(degree=degree)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse,variance, std_dev = calculate_metrics2(y_test, y_pred)
        print(f'Degree {degree} (No Reg) - MSE: {mse}, Std Dev: {std_dev}, Variance: {variance}')
        
        model_l2 = PolynomialRegression(degree=degree, regularization='L2', alpha=alpha)
        model_l2.fit(X_train, y_train)
        y_pred_l2 = model_l2.predict(X_test)
        
        mse_l2, variance_l2,std_dev_l2 = calculate_metrics2(y_test, y_pred_l2)
        print(f'Degree {degree} (L2) - MSE: {mse_l2}, Std Dev: {std_dev_l2}, Variance: {variance_l2}')
        
        model_l1 = PolynomialRegression(degree=degree, regularization='L1', alpha=alpha, learning_rate=learning_rate)
        model_l1.fit(X_train, y_train)
        y_pred_l1 = model_l1.predict(X_test)
        
        mse_l1, variance_l1,std_dev_l1 = calculate_metrics2(y_test, y_pred_l1)
        print(f'Degree {degree} (L1) - MSE: {mse_l1}, Std Dev: {std_dev_l1}, Variance: {variance_l1}')


degrees = list(range(1, 11))

plot_polynomial_regression(X_train, y_train, X_test, y_test, degrees, alpha=0.01, learning_rate=0.001)

# np.open('./assignments/1/best_model_params.npy')

model = PolynomialRegression(degree=best_k)

model.fit(X_train, y_train, path='./assignments/1/best_model_params.npy')


y_test_pred = model.predict(X_test)

test_metrics = calculate_metrics2(y_test, y_test_pred)
results2=[]

results2.append((best_k, *test_metrics))

for result in results2:
    print(f"k = {result[0]}, MSE: {result[1]}, Variance: {result[2]}, Standard Dev: {result[3]}") 




# max_degree = 5
# for k in range(1, max_degree + 1):
#     model = PolynomialRegression(degree=k)
#     model.fit(X_train, y_train,path2='./assignments/1/figures')


k=5
def generate_image_paths(k):
    return [os.path.join("./assignments/1/figures/", f"{k}_iteration_{iteration}.png") for iteration in range(1000)]


image_paths = generate_image_paths(k)

images = [Image.open(image_path) for image_path in image_paths if os.path.exists(image_path)]

gif_path = f"./assignments/1/figures/linreg_{k}.gif"

if images:
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:], 
        duration=5, 
        loop=0  
    )
    print(f"GIF saved to {gif_path}")
else:
    print("No images")




    #LINEAR REGRESSION codes ends here.................................plot code starts.>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns

def load_data(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Drop unnecessary columns
    data = data.drop(['track_id', 'artists', 'album_name', 'track_name'], axis=1)
    data = data.dropna(axis=0)
    
    # Standardize the feature columns (all except the last one)
    feature_columns = data.columns[:-1]
    data[feature_columns] = (data[feature_columns] - data[feature_columns].mean()) / data[feature_columns].std()
    
    # Separate features (X) and target (y)
    X = data.iloc[:, :-1].values  # All columns except the last one are features
    y = data.iloc[:, -1].values   # The last column is the target label (track_genre)
    
    return X, y


csv_file = './data/external/spotify.csv' 

X, y = load_data(csv_file)



data = pd.read_csv(csv_file)
data = data.drop(['track_id', 'artists', 'album_name', 'track_name'], axis=1).dropna(axis=0)
feature_names = data.columns[:-1]

# 1. Histograms
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.ravel()

for i, col in enumerate(feature_names):
    sns.histplot(data[col], ax=axes[i], kde=True)
    axes[i].set_title(col)

plt.tight_layout()
plt.savefig('feature_histograms.png')
plt.close()

# 2. Box plots
plt.figure(figsize=(15, 10))
sns.boxplot(data=data[feature_names])
plt.xticks(rotation=90)
plt.title('Feature Distributions')
plt.tight_layout()
plt.savefig('feature_boxplots.png')
plt.close()

# 3. Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(data[feature_names].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# 4. Pair plot (for selected features)
selected_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness']
sns.pairplot(data[selected_features + ['track_genre']], hue='track_genre', plot_kws={'alpha': 0.5})
plt.tight_layout()
plt.savefig('pairplot.png')
plt.close()

# 5. Genre distribution
plt.figure(figsize=(12, 6))
data['track_genre'].value_counts().plot(kind='bar')
plt.title('Genre Distribution')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('genre_distribution.png')
plt.close()



#PLOT codes ends here...........................................................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>