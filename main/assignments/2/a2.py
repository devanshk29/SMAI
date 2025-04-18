
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from sklearn.datasets import make_blobs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.k_means.k_means import KMeans
df = pd.read_feather('./data/external/word-embeddings.feather')
X = []  
for elem in df["vit"]:
    X.append(elem)
X=np.array(X)
print(X.shape)

# df=pd.read_csv("./assignments/2/data.csv")
# X=np.array(df[["x","y"]])

max_k = 10
wcss = []

for k in range(1, max_k + 1):
    kmeans = KMeans(k)
    kmeans.fit(X)
    # print(f"WCSS for k={k}: {kmeans.getCost()}")
    wcss.append(kmeans.getCost())

# Plot the Elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), wcss, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
# plt.savefig("./assignments/2/figures/3a.png")
plt.show()

k_optimal = 2 #form observation of the elbow curve

print(f"The optimal number of clusters (k_means1) is: {k_optimal}")

kmeans_optimal = KMeans(k_optimal)
kmeans_optimal.fit(X)
for i in range(k_optimal):
    points_in_cluster = X[kmeans_optimal.labels == i]
    print(f"Cluster {i + 1}:")
    print(points_in_cluster)
    print("\n")


print(f"Final WCSS for k={k_optimal}: {kmeans_optimal.getCost()}")

#GMM inbuit
#############################################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    if covariance.shape == (2, 2):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)
    else:
        angle = 0
        width = height = 2 * np.sqrt(covariance)

    ellipse = Ellipse(xy=position, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

# df = pd.read_csv("./assignments/2/data.csv")
# X = np.array(df[["x", "y"]])


df = pd.read_feather('./data/external/word-embeddings.feather')
X = np.array([elem for elem in df["vit"]])

max_clusters = 10
bic_values = []
aic_values = []

for n_clusters in range(1, max_clusters + 1):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(X)
    
    bic_values.append(gmm.bic(X))
    aic_values.append(gmm.aic(X))

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters + 1), bic_values, label='BIC', marker='o')
plt.plot(range(1, max_clusters + 1), aic_values, label='AIC', marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('BIC / AIC score')
plt.title('BIC and AIC for GMM inbuilt')
plt.savefig("./assignments/2/figures/4a.png")
plt.legend()
# plt.show()

optimal_k_bic = np.argmin(bic_values) + 1  
optimal_k_aic = np.argmin(aic_values) + 1  

print(f"Optimal number of clusters (based on BIC): {optimal_k_bic}")
print(f"Optimal number of clusters (based on AIC): {optimal_k_aic}")

# k_gmm_optimal = optimal_k_bic  
# gmm_optimal = GaussianMixture(n_components=k_gmm_optimal, random_state=42)
# gmm_optimal.fit(X)

# # Get cluster labels for the data points
# cluster_labels = gmm_optimal.predict(X)

# plt.figure(figsize=(8, 6))
# ax = plt.gca()

# for cluster in range(k_gmm_optimal):
#     points_in_cluster = X[cluster_labels == cluster]
#     plt.scatter(points_in_cluster[:, 0], points_in_cluster[:, 1], label=f'Cluster {cluster + 1}')
    
#     # Draw an ellipse representing the covariance of the cluster
#     draw_ellipse(gmm_optimal.means_[cluster], gmm_optimal.covariances_[cluster], ax, 
#                  edgecolor='black', facecolor='none', linewidth=2)

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('GMM Clustering with Ellipses')
# plt.legend()
# plt.show()

#GMM self
#############################################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.patches import Ellipse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.gmm.gmm import GMM as GaussianMixtureModel
def draw_ellipse(position, covariance, ax, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    if covariance.shape == (2, 2):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)
    else:
        angle = 0
        width = height = 2 * np.sqrt(covariance)

    ellipse = Ellipse(xy=position, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

# df = pd.read_csv("./assignments/2/data.csv")
# X = np.array(df[["x", "y"]])

df = pd.read_feather('./data/external/word-embeddings.feather')
X = np.array([elem for elem in df["vit"]])

max_clusters = 5
bic_values = []
aic_values = []

for n_clusters in range(1, max_clusters + 1):
    print(f"Trying {n_clusters} clusters")
    gmm = GaussianMixtureModel(n_components=n_clusters, max_iter=100, tol=1e-3)
    gmm.fit(X)
    
    log_likelihood = gmm.getLikelihood(X)
    num_params = n_clusters * (X.shape[1] * (X.shape[1] + 1) / 2 + X.shape[1] + 1) 
    bic = -2 * log_likelihood + num_params * np.log(X.shape[0])
    aic = -2 * log_likelihood + 2 * num_params
    
    bic_values.append(bic)
    aic_values.append(aic)

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters + 1), bic_values, label='BIC', marker='o')
plt.plot(range(1, max_clusters + 1), aic_values, label='AIC', marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('BIC / AIC score')
plt.title('BIC and AIC for GMM self made')
plt.legend()
# plt.savefig("./assignments/2/figures/4b.png")
plt.show()

optimal_k_bic = np.argmin(bic_values) + 1  
optimal_k_aic = np.argmin(aic_values) + 1  

print(f"Optimal number of clusters (based on BIC): {optimal_k_bic}")
print(f"Optimal number of clusters (based on AIC): {optimal_k_aic}")





#PCA
#############################################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from models.pca.pca import PCA 


df = pd.read_feather('./data/external/word-embeddings.feather')

X = np.array(df['vit'].to_list())  
print(f"Original data shape: {X.shape}")



pca_2d = PCA(n_components=2)
pca_2d.fit(X)
X_2d = pca_2d.transform(X)

print( pca_2d.checkPCA(X)), "PCA for 2D"

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c='blue', edgecolor='k')
plt.title('PCA Projection to 2D')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
# plt.savefig("./assignments/2/figures/5a.png")
plt.show()

pca_3d = PCA(n_components=3)
pca_3d.fit(X)
X_3d = pca_3d.transform(X)

print( pca_3d.checkPCA(X)), "PCA for 2D"

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c='blue', edgecolor='k')



ax.set_title('PCA Projection to 3D')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set
# plt.savefig("./assignments/2/figures/5b.png")
plt.show()


print("After observing the plots, the number of cluster looks to be 3")

#Q6 for KMEANS
#############################################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from models.pca.pca import PCA
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.k_means.k_means import KMeans

df = pd.read_feather('./data/external/word-embeddings.feather')

X = np.array(df['vit'].to_list())  
print(f"Original data shape: {X.shape}")



pca_2d = PCA(n_components=2)
pca_2d.fit(X)
X_2d = pca_2d.transform(X)

k = 3  
kmeans = KMeans(k=k)
kmeans.fit(X_2d)

labels = kmeans.labels

centroids = kmeans.centroids

plt.figure(figsize=(10, 7))
for i in range(k):
    plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1], label=f"Cluster {i+1}")

plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')

plt.title(f"K-means Clustering with {k} Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
# plt.savefig("./assignments/2/figures/6a.png")
# plt.show()



df = pd.read_feather('./data/external/word-embeddings.feather')

X = np.array(df['vit'].to_list())

pca_all = PCA(n_components=X.shape[1])
pca_all.fit(X)

X_centered = X - np.mean(X, axis=0)
cov_matrix = np.cov(X_centered, rowvar=False)
eigenvalues, _ = np.linalg.eig(cov_matrix)
eigenvalues = np.real(eigenvalues)

explained_variance = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Cumulative Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)

plt.tight_layout()
# plt.savefig("./assignments/2/figures/6b.png")


n_optimal = 2
pca_optimal = PCA(n_components=n_optimal)
pca_optimal.fit(X)
X_reduced = pca_optimal.transform(X)

print(pca_optimal.checkPCA(X))
print(f"Reduced dataset shape: {X_reduced.shape}")


wcss = []
cluster_range = range(1, 11)  

for k in cluster_range:
    kmeans = KMeans(k=k)
    kmeans.fit(X_reduced)
    wcss.append(kmeans.getCost())

plt.figure(figsize=(10, 6))
plt.plot(cluster_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
# plt.savefig("./assignments/2/figures/6c.png")
# plt.show()

kkmeans3 = 2  


kmeans_optimal = KMeans(k=kkmeans3)
kmeans_optimal.fit(X_reduced)

labels = kmeans_optimal.labels
centroids = kmeans_optimal.centroids

if n_optimal == 2:
    plt.figure(figsize=(10, 7))
    for i in range(kkmeans3):
        plt.scatter(X_reduced[labels == i, 0], X_reduced[labels == i, 1], label=f"Cluster {i+1}")

    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
    plt.title(f"K-means Clustering with {kkmeans3} Clusters (2D PCA Reduced Data)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    # plt.show()
    # plt.savefig("./assignments/2/figures/6d.png")

# Q6 for GMM
#############################################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from models.pca.pca import PCA  
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.gmm.gmm import GMM as GaussianMixtureModel

df = pd.read_feather('./data/external/word-embeddings.feather')

X = np.array(df['vit'].to_list())  
print(f"Original data shape: {X.shape}")



pca_2d = PCA(n_components=2)
pca_2d.fit(X)
X_2d = pca_2d.transform(X)

k = 3
gmm = GaussianMixtureModel(n_components=k)
gmm.fit(X_2d)

cluster_assignments = gmm.predict(X_2d)

w,means,covariances = gmm.getParams()

plt.figure(figsize=(10, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_assignments, cmap='viridis', marker='o', edgecolor='k', alpha=0.7)

for i in range(k):
    mean = means[i]
    covariance = covariances[i]
    v, w = np.linalg.eigh(covariance)
    u = w[:, np.argmax(v)]
    v = np.sqrt(v[np.argmax(v)])
    t = np.linspace(0, 2 * np.pi, 100)
    x = v * np.cos(t)
    y = v * np.sin(t)
    ellipse = np.dot(np.vstack([x, y]).T, w) + mean
    plt.plot(ellipse[:, 0], ellipse[:, 1], color='k')

plt.title('GMM Clustering with k=3')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
# plt.savefig("./assignments/2/figures/6_b1.png")
plt.show()


n_optimal = 2 
pca_optimal = PCA(n_components=n_optimal)
pca_optimal.fit(X)
X_optimal = pca_optimal.transform(X)



max_clusters = 10
bic_values = []
aic_values = []

for n_clusters in range(1, max_clusters + 1):
    # print(f"Trying {n_clusters} clusters")
    gmm = GaussianMixtureModel(n_components=n_optimal)
    gmm.fit(X_optimal)
    
    log_likelihood = gmm.getLikelihood(X_optimal)
    num_params = n_clusters * (X_optimal.shape[1] * (X_optimal.shape[1] + 1) / 2 + X_optimal.shape[1] + 1) 
    bic = -2 * log_likelihood + num_params * np.log(X_optimal.shape[0])
    aic = -2 * log_likelihood + 2 * num_params
    
    bic_values.append(bic)
    aic_values.append(aic)

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters + 1), bic_values, label='BIC', marker='o')
plt.plot(range(1, max_clusters + 1), aic_values, label='AIC', marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('BIC / AIC score')
plt.title('BIC and AIC for GMM self made')
plt.legend()
# plt.savefig("./assignments/2/figures/6_b2.png")
plt.show()

optimal_k_bic = np.argmin(bic_values) + 1 
optimal_k_aic = np.argmin(aic_values) + 1  

print(f"Optimal number of clusters (based on BIC): {optimal_k_bic}")
print(f"Optimal number of clusters (based on AIC): {optimal_k_aic}")


k= 3

gmm = GaussianMixtureModel(n_components=k)
gmm.fit(X_optimal)

cluster_assignments = gmm.predict(X_optimal)

w,means,covariances = gmm.getParams()

plt.figure(figsize=(10, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_assignments, cmap='viridis', marker='o', edgecolor='k', alpha=0.7)

for i in range(k):
    mean = means[i]
    covariance = covariances[i]
    v, w = np.linalg.eigh(covariance)
    u = w[:, np.argmax(v)]
    v = np.sqrt(v[np.argmax(v)])
    t = np.linspace(0, 2 * np.pi, 100)
    x = v * np.cos(t)
    y = v * np.sin(t)
    ellipse = np.dot(np.vstack([x, y]).T, w) + mean
    plt.plot(ellipse[:, 0], ellipse[:, 1], color='k')

plt.title('GMM Clustering with k=8')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
# plt.savefig("./assignments/2/figures/6_b4.png")
plt.show()

df['cluster'] = cluster_assignments

for cluster_num in range(k):
    print(f"Cluster {cluster_num}:")
    cluster_words = df[df['cluster'] == cluster_num]['words']
    print(cluster_words.to_list())
    print()


#Q7
#############################################################################################################################################


import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.k_means.k_means import KMeans


df = pd.read_feather('./data/external/word-embeddings.feather')

X = np.array(df["vit"].tolist())  

y = df["words"].values

k_optimal = 2

kmeans_optimal = KMeans(k_optimal)
kmeans_optimal.fit(X)

for i in range(k_optimal):
    cluster_indices = np.where(kmeans_optimal.labels == i)[0]
    words_in_cluster = y[cluster_indices]
    print(f"Cluster {i + 1}:")
    print(words_in_cluster)
    print("\n")

print(f"Final WCSS for k={k_optimal}: {kmeans_optimal.getCost()}")


#Q8 hierarchical clustering
#############################################################################################################################################


#Q8 hierarchical clustering
#############################################################################################################################################

import numpy as np
import scipy.cluster.hierarchy as hc
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import fcluster

df = pd.read_feather('./data/external/word-embeddings.feather')
X = np.array([elem for elem in df["vit"]])
words = df["words"].values
linkage_methods = ['complete', 'average', 'median']

kbest1 = 3  
kbest2 = 3  

for method in linkage_methods:
    Z = hc.linkage(X, method=method, metric='cityblock')
    
    # Plot dendrogram
    plt.figure(figsize=(10, 7))
    hc.dendrogram(Z)
    plt.title(f"Dendrogram - {method.capitalize()} Linkage")
    plt.savefig(f"./assignments/2/figures/8_{method}.png")
    # plt.show()

method="ward"
kbest1 = 3  
kbest2 = 3

Z = hc.linkage(X, method="ward", metric='euclidean')
clusters_kbest1 = fcluster(Z, kbest1, criterion='maxclust')
clusters_kbest2 = fcluster(Z, kbest2, criterion='maxclust')

plt.figure(figsize=(12, 6))

plt.scatter(X[:, 0], X[:, 1], c=clusters_kbest1, cmap='viridis')
plt.title(f"Clusters for k={kbest1} ({method.capitalize()} Linkage)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
# plt.savefig(f"./assignments/2/figures/8_{method}_kbest1.png")
# plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters_kbest2, cmap='plasma')
plt.title(f"Clusters for k={kbest2} ({method.capitalize()} Linkage)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
# plt.savefig(f"./assignments/2/figures/8_{method}_kbest2.png")
# plt.show()


clustered_df = pd.DataFrame({
    'word': words,
    'cluster': clusters_kbest1
})

for cluster_num, group in clustered_df.groupby('cluster'):
    print(f"Cluster {cluster_num}:")
    print(', '.join(group['word']))
    print()

#Q9 PCA with KNN
#############################################################################################################################################



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.pca.pca import PCA
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.knn import knn as KNN
import time



def load_data(csv_file):
    data = pd.read_csv(csv_file)
    
    data = data.drop(['track_id', 'artists', 'album_name', 'track_name', 'key', 'mode', 'liveness', 'valence'], axis=1)
    data = data.dropna(axis=0)
    
    feature_columns = data.columns[:-1]
    data[feature_columns] = (data[feature_columns] - data[feature_columns].mean()) / data[feature_columns].std()
    
    X = data.iloc[:, :-1].values  
    y = data.iloc[:, -1].values   
    return X, y

def split_data(X, y, train_ratio=0.8, validate_ratio=0.0100):
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



csv_file = './data/external/spotify.csv'  
X, y = load_data(csv_file)

X_train, X_validate, X_test, y_train, y_validate, y_test = split_data(X, y)

# run_knn(csv_file, k=28, distance_metric='manhattan', run_calc='test')


pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)
pca.transform(X_train)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1) 
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'ro-')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot (Explained Variance)')
plt.grid(True)
# plt.show()
# plt.savefig("./assignments/2/figures/9a.png")

plt.subplot(1, 2, 2) 
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Scree Plot (Cumulative Explained Variance)')
# plt.show()
plt.grid(True)
# plt.savefig("./assignments/2/figures/9b.png")




optimal_components = np.argmax(cumulative_variance_ratio >= 0.90) + 1
print(f"Optimal number of components: {optimal_components}")

pca_optimal = PCA(n_components=optimal_components)
pca_optimal.fit(X_train)
X_train_reduced = pca_optimal.transform(X_train)

pca_optimal_test = PCA(n_components=optimal_components)
pca_optimal_test.fit(X_test)
X_test_reduced = pca_optimal_test.transform(X_test)


best_k = 28
best_metric = 'manhattan'
best_k1, best_metric1, path='./assignments/1/best_model_params.npy'

knn = KNN.KNN(k=best_k, distance_metric=best_metric)
knn.fit(X_train_reduced, y_train)

y_pred = knn.predict(X_test_reduced)
metrics = KNN.Metrics(knn.label_mapping)

accuracy = metrics.accuracy(y_test, y_pred)

precision_macro = metrics.precision(y_test, y_pred, average='macro')
recall_macro = metrics.recall(y_test, y_pred, average='macro')
f1_macro = metrics.f1_score(y_test, y_pred, average='macro')

precision_micro = metrics.precision(y_test, y_pred, average='micro')
recall_micro = metrics.recall(y_test, y_pred, average='micro')
f1_micro = metrics.f1_score(y_test, y_pred, average='micro')

print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Precision (Macro): {precision_macro * 100:.2f}%")
print(f"Validation Recall (Macro): {recall_macro * 100:.2f}%")
print(f"Validation F1 Score (Macro): {f1_macro * 100:.2f}%")
print(f"Validation Precision (Micro): {precision_micro * 100:.2f}%")
print(f"Validation Recall (Micro): {recall_micro * 100:.2f}%")
print(f"Validation F1 Score (Micro): {f1_micro * 100:.2f}%")

assignment1_accuracy = 56.32
assignment1_precision_macro = 57.33
assignment1_precision_micro = 56.32
assignment1_recall_macro = 55.91
assignment1_recall_micro = 56.32
assignment1_f1_macro = 56.61
assignment1_f1_micro = 56.32

# Validation Accuracy: 56.32%
# Validation Precision (Macro): 57.33%
# Validation Recall (Macro): 55.91%
# Validation F1 Score (Macro): 56.61%
# Validation Precision (Micro): 56.32%
# Validation Recall (Micro): 56.32%
# Validation F1 Score (Micro): 56.32%


print("\nComparison with Assignment 1 results:")
print(f"Accuracy: {accuracy*100:.4f} vs {assignment1_accuracy:.4f}")   
print(f"Precision (Macro): {precision_macro*100:.4f} vs {assignment1_precision_macro:.4f}")
print(f"Recall (Macro): {recall_macro*100:.4f} vs {assignment1_recall_macro:.4f}")
print(f"F1 Score (Macro): {f1_macro*100:.4f} vs {assignment1_f1_macro:.4f}")
print(f"Precision (Micro): {precision_micro*100:.4f} vs {assignment1_precision_micro:.4f}")
print(f"Recall (Micro): {recall_micro*100:.4f} vs {assignment1_recall_micro:.4f}")
print(f"F1 Score (Micro): {f1_micro*100:.4f} vs {assignment1_f1_micro:.4f}")


def measure_inference_time(model, X, num_runs=5):
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        model.predict(X)
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times)

time_full = measure_inference_time(knn, X_test)
time_reduced = measure_inference_time(knn, X_test_reduced)

plt.figure(figsize=(8, 6))
plt.bar(['Full Dataset', 'Reduced Dataset'], [time_full, time_reduced])
plt.ylabel('Inference Time (seconds)')
plt.title('KNN Inference Time Comparison')
# plt.show()
# plt.savefig("./assignments/2/figures/9c.png")

print(f"\nInference time (Full Dataset): {time_full:.4f} seconds")
print(f"Inference time (Reduced Dataset): {time_reduced:.4f} seconds")


#############################################################################################################################################