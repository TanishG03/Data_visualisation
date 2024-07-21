import numpy as np
import pandas as pd
from itertools import chain, combinations
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import io
from PIL import Image

def powerset(s):
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))

def compute_knn(X, k, subspace_indices):
    """Compute k-nearest neighbors for each point in the specified subspace."""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X[:, subspace_indices])
    distances, indices = nbrs.kneighbors(X[:, subspace_indices])
    return indices

def heidi_matrix_single_dimension(X, dim, k):
    """Compute the Heidi matrix for a single dimension."""
    n = X.shape[0]
    H = np.zeros((n, n), dtype=int)

    knn_indices = compute_knn(X, k, [dim])

    for i in range(n):
        for j in knn_indices[i]:
            H[i, j] = 1

    return H

def heidi_matrix(X, D, k):
    n = X.shape[0]
    P = powerset(D)
    subspace_map = {subspace: idx for idx, subspace in enumerate(P)}
    H = np.zeros((n, n, len(P)), dtype=int)

    for subspace in P:
        subspace_indices = list(subspace)
        knn_indices = compute_knn(X, k, subspace_indices)

        for i in range(n):
            for j in knn_indices[i]:
                H[i, j, subspace_map[subspace]] = 1

    return H

def knn_ordering(knn_indices):
    visited = set()
    order = []

    def dfs(node):
        if node not in visited:
            visited.add(node)
            order.append(node)
            for neighbor in knn_indices[node]:
                dfs(neighbor)

    for start_node in range(knn_indices.shape[0]):
        if start_node not in visited:
            dfs(start_node)

    return order

def save_and_encode_image(fig):
    """Save the given figure to a bytes buffer and encode it as an image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_data = buf.getvalue()
    return image_data


def visualize_heidi_matrix_single(H, cluster_labels, X, dim, k):
    sorted_indices = np.argsort(cluster_labels)
    cluster_labels_sorted = cluster_labels[sorted_indices]

    final_order = []
    unique_labels = np.unique(cluster_labels_sorted)

    for label in unique_labels:
        cluster_indices = np.where(cluster_labels_sorted == label)[0]
        knn_indices = compute_knn(X[sorted_indices[cluster_indices]], k, [dim])
        order = knn_ordering(knn_indices)
        final_order.extend(cluster_indices[order])

    # print(final_order)

    H_reordered = H[sorted_indices[final_order]][:, sorted_indices[final_order]]

    n = H.shape[0]

    # Create a custom colormap that starts with white
    cmap = mcolors.LinearSegmentedColormap.from_list('white_to_viridis', ['white', 'blue', 'green', 'yellow', 'red'])

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(H_reordered, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Presence in k-NN')
    plt.title(f'Heidi Matrix Visualization for Dimension {dim} (kNN Ordering)')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    image_data = save_and_encode_image(fig)
    return image_data

def visualize_combined_heidi_matrix(H_list, cluster_labels, X, k):
    # Combine the individual Heidi matrices
    H_combined = np.sum(H_list, axis=0)
    
    sorted_indices = np.argsort(cluster_labels)
    H_sorted = H_combined[sorted_indices][:, sorted_indices]
    cluster_labels_sorted = cluster_labels[sorted_indices]
    unique_labels = np.unique(cluster_labels_sorted)

    final_order = []
    for label in unique_labels:
        cluster_indices = np.where(cluster_labels_sorted == label)[0]
        knn_indices = compute_knn(X[sorted_indices[cluster_indices]], k, list(range(X.shape[1])))
        order = knn_ordering(knn_indices)
        final_order.extend(cluster_indices[order])

    H_reordered = H_sorted[final_order][:, final_order]

    n = H_combined.shape[0]
    P_len = H_combined.shape[2]

    H_int = np.sum(H_reordered * (2 ** np.arange(P_len)[::-1]), axis=2)

    H_max = np.max(H_int)
    H_norm = H_int / H_max if H_max != 0 else H_int  # Avoid division by zero
    H_norm = np.clip(H_norm, 0, 1)  # Ensure values are between 0 and 1

    # Define a custom colormap with shades of blue
    cmap = mcolors.LinearSegmentedColormap.from_list('shades_of_blue', [(0, 'white'), (0.5, 'blue'), (1, 'navy')])

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(H_norm, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Normalized Bit Vector Value')
    plt.title('Combined Heidi Matrix Visualization (kNN Ordering within Clusters)')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    image_data = save_and_encode_image(fig)
    return image_data

def visualize_heidi_matrix(H, cluster_labels, X, k):
    sorted_indices = np.argsort(cluster_labels)
    H_sorted = H[sorted_indices][:, sorted_indices]
    cluster_labels_sorted = cluster_labels[sorted_indices]
    unique_labels = np.unique(cluster_labels_sorted)

    final_order = []
    for label in unique_labels:
        cluster_indices = np.where(cluster_labels_sorted == label)[0]
        knn_indices = compute_knn(X[sorted_indices[cluster_indices]], k, list(range(X.shape[1])))
        order = knn_ordering(knn_indices)
        final_order.extend(cluster_indices[order])


    H_reordered = H_sorted[final_order][:, final_order]

    n = H.shape[0]
    P_len = H.shape[2]

    H_int = np.sum(H_reordered * (2 ** np.arange(P_len)[::-1]), axis=2)

    H_max = np.max(H_int)
    H_norm = H_int / H_max
    H_norm = H_int / H_max if H_max != 0 else H_int  # Avoid division by zero
    H_norm = np.clip(H_norm, 0, 1)  # Ensure values are between 0 and 1

    # Define a custom colormap with shades of blue
    cmap = mcolors.LinearSegmentedColormap.from_list('shades_of_blue', [(0, 'white'), (0.5, 'blue'), (1, 'navy')])

    fig = plt.figure(figsize=(10, 8))
    
    # Overlay the colors with alpha values
    plt.imshow(H_norm, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Set transparency based on normalized score
    plt.imshow(H_norm, cmap=cmap, aspect='auto', vmin=0, vmax=1, alpha=H_norm)
    
    plt.colorbar(label='Normalized Bit Vector Value')
    plt.title('Heidi Matrix Visualization (kNN Ordering within Clusters)')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    # plt.show()
    image_data = save_and_encode_image(fig)
    return image_data




# Load the dataset, ignoring the first and last columns
# data = pd.read_csv('../Iris.csv').iloc[:, 1:-1]

def main(filepath):


    _, file_extension = os.path.splitext(filepath)
        
        # Check if the file is iris.csv
    if os.path.basename(filepath) == 'Iris.csv':
            data = pd.read_csv(filepath).iloc[:, 1:-1]
    elif file_extension.lower() == '.csv':
            data = pd.read_csv(filepath).iloc[:, :-1]
    elif file_extension.lower() in ('.xls', '.xlsx'):
            data = pd.read_excel(filepath).iloc[:, :-1]
    else:
            raise ValueError("Unsupported file format")


    label_encoders = {}
    for column in data.select_dtypes(include=['object']):
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(scaled_data)
    cluster_labels = kmeans.labels_

    # Define dimensions D (all columns)
    D = range(scaled_data.shape[1])
    k = 10

    H = heidi_matrix(scaled_data, D, k)

    # visualize_heidi_matrix(H, cluster_labels, scaled_data, k)

    H_list = []
    image_data_list = []
    # Visualize each single-dimensional Heidi matrix and store in the list
    for dim in D:
        H_single = heidi_matrix_single_dimension(scaled_data, dim, k)
        H_list.append(H_single)
        image_data=visualize_heidi_matrix_single(H_single, cluster_labels, scaled_data, dim, k)
        image_data_list.append(image_data)
        # print(len(image_data_list))

    # print(len(image_data_list))
    return {"data": cluster_labels.tolist()}, image_data_list

    # Visualize the combined Heidi matrix
    # visualize_combined_heidi_matrix(H_list, cluster_labels, scaled_data, k)