from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from itertools import chain, combinations
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import os
import io
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import minimum_spanning_tree

def powerset(s):
    """Generate all non-empty subsets of a set."""
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))

def compute_knn(X, k, subspace_indices):
    """Compute k-nearest neighbors for each point in the specified subspace."""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X[:, subspace_indices])
    distances, indices = nbrs.kneighbors(X[:, subspace_indices])
    return indices

def heidi_matrix(X, D, k):
    """Compute the HEIDI matrix using k-nearest neighbors."""
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

    return H, P

def knn_ordering(knn_indices):
    """Determine the order of points based on k-nearest neighbors using DFS."""
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

def spectral_ordering(knn_indices):
    """Determine the order of points based on spectral ordering using the Laplacian of the kNN graph."""
    # print("hi")
    n = knn_indices.shape[0]
    adjacency_matrix = np.zeros((n, n))

    # Build adjacency matrix from kNN indices
    for i in range(n):
        for j in knn_indices[i]:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1  # Ensure symmetry

    # Compute the graph Laplacian
    lap = laplacian(adjacency_matrix, normed=True)

    # Compute the eigenvalues and eigenvectors of the Laplacian
    _, eigvecs = eigsh(lap, k=2, which='SM')
    fiedler_vector = eigvecs[:, 1]

    # Order points by the Fiedler vector
    order = np.argsort(fiedler_vector)
    print(order)

    return order

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

def hierarchical_ordering(X, method='ward'):
    """Determine the order of points using hierarchical clustering."""
    # Compute pairwise distances
    dist_matrix = pdist(X)
    
    # Perform hierarchical clustering
    Z = linkage(dist_matrix, method=method)
    
    # Cut the dendrogram to get cluster labels
    labels = fcluster(Z, t=X.shape[0], criterion='maxclust')
    
    # Sort points based on cluster labels
    order = np.argsort(labels)
    
    return order


from sklearn.manifold import TSNE

def tsne_ordering(X):
    """Generate ordering based on t-SNE."""
    tsne = TSNE(n_components=2, perplexity=30)
    reduced_X = tsne.fit_transform(X)
    
    # Order by the first dimension of the t-SNE output
    order = np.argsort(reduced_X[:, 0])
    
    return order

from sklearn.cluster import AffinityPropagation

def affinity_propagation_ordering(X):
    """Determine the order of points using Affinity Propagation clustering."""
    aff_propag = AffinityPropagation(random_state=0)
    cluster_labels = aff_propag.fit_predict(X)
    
    # Sort points based on cluster labels
    order = np.argsort(cluster_labels)
    
    return order

from collections import deque

def bfs_ordering(knn_indices):
    """Generate ordering based on BFS traversal of kNN graph."""
    n = knn_indices.shape[0]
    adjacency_matrix = np.zeros((n, n))

    # Build adjacency matrix from kNN indices
    for i in range(n):
        adjacency_matrix[i, knn_indices[i]] = 1
    
    # Ensure symmetry
    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)

    visited = set()
    order = []
    queue = deque([0])  # Start with the first node

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            order.append(node)
            neighbors = np.where(adjacency_matrix[node] != 0)[0]
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

    return order


def save_and_encode_image(fig):
    """Save the given figure to a bytes buffer and encode it as an image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_data = buf.getvalue()
    return image_data

def visualize_all_subspaces_knn(H, P, knn_indices, X):
    """Visualize the HEIDI matrix using kNN ordering for all subspaces."""
    n = H.shape[0]

    # Get the kNN ordering
    order = knn_ordering(knn_indices)
    # order = spectral_ordering(knn_indices)

    # Create a white background with a blue spectrum for the heatmap
    blue_spectrum = plt.cm.Blues

    # Determine grid size for square-like visualization
    grid_size = int(np.ceil(np.sqrt(len(P))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    for idx, subspace in enumerate(P):
        subspace_idx = P.index(subspace)
        H_subspace_reordered = H[:, :, subspace_idx][order][:, order]
        
        # Determine the grid position
        row, col = divmod(idx, grid_size)
        ax = axes[row, col]
        
        # Plot the subspace heatmap
        cax = ax.imshow(H_subspace_reordered, cmap=blue_spectrum, aspect='auto')
        ax.set_title(f"Subspace: {subspace}")
        ax.axis('off')  # Turn off axis labels

    # Hide any unused subplots
    for idx in range(len(P), grid_size * grid_size):
        row, col = divmod(idx, grid_size)
        fig.delaxes(axes[row, col])

    fig.suptitle('HEIDI Matrix Visualization for All Subspaces (kNN Ordering)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Show colorbar on the right of the grid
    fig.colorbar(cax, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)

    plt.show()
    image_data = save_and_encode_image(fig)
    plt.close(fig)  # Close the figure after saving

    return image_data  # Return the image data

from matplotlib.colors import LinearSegmentedColormap
def create_custom_colormap():
    """Create a custom colormap to emphasize value ranges."""
    colors = ["#000000", "#FF4500", "#FFFF00", "#32CD32", "#0000FF"]  # Black, Red, Yellow, Green, Blue
    n_bins = [0.0, 0.25, 0.5, 0.75, 1.0]  # Define intervals for the color spectrum
    cmap_name = "contrast_cmap"
    return LinearSegmentedColormap.from_list(cmap_name, list(zip(n_bins, colors)))

def visualize_combined_subspaces_knn(H, P, knn_indices, X):
    """Visualize a combined HEIDI matrix for all subspaces using kNN ordering with contrasty colors."""
    n = H.shape[0]

    # Get the kNN ordering
    order = knn_ordering(knn_indices)
    # order = bfs_ordering(knn_indices)

    # Aggregate the HEIDI matrix over all subspaces to get a single combined matrix
    H_combined = np.sum(H, axis=2)

    # Reorder the combined matrix based on kNN ordering
    H_combined_reordered = H_combined[order][:, order]

    # Create the custom colormap
    custom_cmap = create_custom_colormap()

    # Plot the combined heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(H_combined_reordered, cmap=custom_cmap, aspect='auto')
    ax.set_title('Combined HEIDI Matrix Visualization (kNN Ordering)')
    ax.axis('off')  # Turn off axis labels

    # Show colorbar
    fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)

    plt.show()
    image_data = save_and_encode_image(fig)
    plt.close(fig)  # Close the figure after saving

    return image_data

# def visualize_combined_subspaces_knn(H, P, knn_indices, X):
#     """Visualize a combined HEIDI matrix for all subspaces using kNN ordering."""
#     n = H.shape[0]

#     # Get the kNN ordering
#     # order = knn_ordering(knn_indices)
#     # order = spectral_ordering(knn_indices)
#     # order = hierarchical_ordering(X)
#     # order = tsne_ordering(X)
#     # order = affinity_propagation_ordering(X)
#     order = bfs_ordering(knn_indices)

#     # Aggregate the HEIDI matrix over all subspaces to get a single combined matrix
#     H_combined = np.sum(H, axis=2)

#     # Reorder the combined matrix based on kNN ordering
#     H_combined_reordered = H_combined[order][:, order]

#     # Create a white background with a blue spectrum for the heatmap
#     blue_spectrum = plt.cm.Blues

#     # Plot the combined heatmap
#     fig, ax = plt.subplots(figsize=(10, 10))
#     cax = ax.imshow(H_combined_reordered, cmap=blue_spectrum, aspect='auto')
#     ax.set_title('Combined HEIDI Matrix Visualization (kNN Ordering)')
#     # ax.axis('off')  # Turn off axis labels

#     # Show colorbar
#     fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)

#     plt.show()
#     image_data = save_and_encode_image(fig)
#     plt.close(fig)  # Close the figure after saving

#     return image_data 

# Example usage
def main(filepath):
    # Load and preprocess your data
    _, file_extension = os.path.splitext(filepath)
    
    if os.path.basename(filepath) == 'Iris.csv':
        data = pd.read_csv(filepath).iloc[:, 1:-1]
    elif file_extension.lower() == '.csv':
        data = pd.read_csv(filepath).iloc[:, :-1]
    elif file_extension.lower() in ('.xls', '.xlsx'):
        data = pd.read_excel(filepath).iloc[:, :-1]
    else:
        raise ValueError("Unsupported file format")

    # Encode categorical data if necessary
    label_encoders = {}
    for column in data.select_dtypes(include=['object']):
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    D = range(scaled_data.shape[1])
    k = 20

    # Compute k-nearest neighbors
    knn_indices = compute_knn(scaled_data, k, list(D))
    
    # Compute HEIDI matrix
    H, P = heidi_matrix(scaled_data, D, k)

    # Visualize all subspaces
    image_data = visualize_combined_subspaces_knn(H, P, knn_indices, scaled_data)

    return {"data": knn_indices.tolist()}, {"visualization": image_data}



def generate_clusters(cluster_centers, cluster_std, n_samples=200, random_state=None):
    """Generate synthetic clusters with specified centers and standard deviations."""
    X, y = make_blobs(n_samples=n_samples, centers=cluster_centers, cluster_std=cluster_std, random_state=random_state)
    return X, y

# Define cluster configurations for three scenarios
cluster_configs = [
    {"centers": [[0, 0], [1, 1]], "std": 1.5, "title": "Highly Overlapping Clusters"},
    {"centers": [[0, 0], [4, 4]], "std": 1.0, "title": "Moderately Overlapping Clusters"},
    {"centers": [[0, 0], [10, 10]], "std": 0.5, "title": "Non-Overlapping Clusters"}
]

# Loop through each configuration and process
for config in cluster_configs:
    # Generate synthetic data
    X, y = generate_clusters(config["centers"], config["std"], random_state=42)
    
    # Plot the generated data
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30, alpha=0.8)
    plt.title(config["title"])
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Standardize the data
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)
    
    # Define the feature dimensions and neighbors
    D = range(scaled_X.shape[1])
    k = 10

    # Compute k-nearest neighbors
    knn_indices = compute_knn(scaled_X, k, list(D))
    
    # Compute HEIDI matrix
    H, P = heidi_matrix(scaled_X, D, k)
    
    # Visualize the combined HEIDI matrix
    print(f"Visualizing HEIDI matrix for {config['title']}")
    visualize_combined_subspaces_knn(H, P, knn_indices, scaled_X)
