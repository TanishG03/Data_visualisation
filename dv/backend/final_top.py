import numpy as np
import pandas as pd
from itertools import chain, combinations
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import io
import os

def compute_cluster_radius(cluster_center, cluster_points):
    distances = euclidean_distances(cluster_points, cluster_center.reshape(1, -1))
    return np.max(distances)

def compute_cluster_diameter(cluster_points):
    distances = euclidean_distances(cluster_points)
    return np.max(distances)

def compute_cluster_quality(cluster_centers, cluster_points):
    cluster_radii = [compute_cluster_radius(center, points) for center, points in zip(cluster_centers, cluster_points)]
    cluster_diameters = [compute_cluster_diameter(points) for points in cluster_points]
    avg_diameter = np.mean(cluster_diameters)
    avg_radius = np.mean(cluster_radii)
    return avg_diameter, avg_radius

def powerset(s):
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))

def compute_knn(X, k, subspace_indices):
    """Compute k-nearest neighbors for each point in the specified subspace."""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X[:, subspace_indices])
    distances, indices = nbrs.kneighbors(X[:, subspace_indices])
    return indices

def heidi_matrix(X, D, k):
    n = X.shape[0]
    P = powerset(D)
    subspace_map = {subspace: idx for idx, subspace in enumerate(P)}
    H = np.zeros((n, n, len(P)), dtype=int)
    subspace_quality = {subspace: 0 for subspace in P}

    for subspace in P:
        subspace_indices = list(subspace)
        knn_indices = compute_knn(X, k, subspace_indices)

        # Compute cluster quality in the current subspace
        kmeans = KMeans()
        kmeans.fit(X[:, subspace_indices])
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
        cluster_points = [X[:, subspace_indices][cluster_labels == i] for i in range(len(cluster_centers))]
        cluster_quality = [compute_cluster_quality([center], [points]) for center, points in zip(cluster_centers, cluster_points)]
        
        # Compute average quality of clusters weighted by the number of points
        total_points = sum(len(points) for points in cluster_points)
        avg_quality = sum((len(points) / total_points) * quality[0] for quality, points in zip(cluster_quality, cluster_points))
        subspace_quality[subspace] = 1 / avg_quality  # Inverse of average diameter for quality

        for i in range(n):
            for j in knn_indices[i]:
                H[i, j, subspace_map[subspace]] = 1

    return H, P, subspace_quality

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

def sort_subspaces_by_cluster_quality(subspace_quality):
    return sorted(subspace_quality.items(), key=lambda x: x[1], reverse=True)

def visualize_top_subspaces(H, P, subspace_quality, cluster_labels, X, k, top_n=10):
    n = H.shape[0]

    # Sort the subspaces by their quality
    sorted_subspaces = sort_subspaces_by_cluster_quality(subspace_quality)
    top_subspaces = [subspace for subspace, _ in sorted_subspaces[:top_n]]

    # Visualize the top subspaces
    combined_matrix = np.zeros((n, n, 3))  # RGB image
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (0.5, 0, 0.5), (0.5, 0.5, 0), (0, 0.5, 0.5)]  # Define unique colors

    sorted_indices = np.argsort(cluster_labels)
    cluster_labels_sorted = cluster_labels[sorted_indices]
    unique_labels = np.unique(cluster_labels_sorted)

    final_order = []
    for label in unique_labels:
        cluster_indices = np.where(cluster_labels_sorted == label)[0]
        knn_indices = compute_knn(X[sorted_indices[cluster_indices]], k, list(range(X.shape[1])))
        order = knn_ordering(knn_indices)
        final_order.extend(cluster_indices[order])

    for i, subspace in enumerate(top_subspaces):
        subspace_idx = P.index(subspace)
        H_subspace_reordered = H[:, :, subspace_idx][sorted_indices[final_order]][:, sorted_indices[final_order]]
        for c in range(3):
            combined_matrix[:, :, c] += H_subspace_reordered * colors[i % len(colors)][c]

    combined_matrix = np.clip(combined_matrix, 0, 1)  # Ensure values are between 0 and 1

    plt.figure(figsize=(10, 8))
    plt.imshow(combined_matrix, aspect='auto')
    plt.title(f'Top {top_n} Subspaces Visualization (kNN Ordering within Clusters)')
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    # Create a legend for the colors
    patches = [plt.plot([], [], marker="s", ms=10, ls="", mec=None, color=colors[i % len(colors)], 
                label="" + str(top_subspaces[i]))[0] for i in range(top_n)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

# Example usage
def main(filepath):
    # Load and preprocess your data
    # data = pd.read_csv(filepath).iloc[:, 1:-1]
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

    kmeans = KMeans()
    kmeans.fit(scaled_data)
    cluster_labels = kmeans.labels_


    D = range(scaled_data.shape[1])
    k = 6

    H, P, subspace_quality = heidi_matrix(scaled_data, D, k)

    # Visualize the top 10 subspaces (you can change the number of top subspaces by modifying the top_n parameter)
    visualize_top_subspaces(H, P, subspace_quality, cluster_labels, scaled_data, k, top_n=10)
    # img_buffer = io.BytesIO()
    # fig.savefig(img_buffer, format='png')
    # img_buffer.seek(0)
    # img_data = img_buffer.getvalue()
    # # plt.close(fig)  # Close the figure to prevent it from being displayed

    # return {'output_data': 'Replace this with your output data', 'image_data': img_data}
