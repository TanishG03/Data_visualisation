import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from itertools import chain, combinations
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

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

    _, eigvecs = eigsh(lap, k=2, which='SM')
    fiedler_vector = eigvecs[:, 1]

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

# do dbscan ordering
from sklearn.cluster import DBSCAN
def dbscan_ordering(X):
    dbscan = DBSCAN(eps=0.3, min_samples=10)
    cluster_labels = dbscan.fit_predict(X)
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

from matplotlib.colors import LinearSegmentedColormap

def create_custom_colormap():
    """Create a custom colormap to emphasize value ranges."""
    colors = ["#000000", "#FF4500", "#FFFF00", "#32CD32", "#0000FF"]  # Black, Red, Yellow, Green, Blue
    n_bins = [0.0, 0.25, 0.5, 0.75, 1.0]  # Define intervals for the color spectrum
    cmap_name = "contrast_cmap"
    return LinearSegmentedColormap.from_list(cmap_name, list(zip(n_bins, colors)))

# Create the custom colormap
custom_cmap = create_custom_colormap()

def visualize_clusters_and_heidi(X, y, H, P, knn_indices, title):
    """
    Create a side-by-side visualization of original clusters and HEIDI matrix.

    Parameters:
    - X: Original data points
    - y: Cluster labels
    - H: HEIDI matrix
    - P: Subspace combinations
    - knn_indices: k-nearest neighbors indices
    - title: Title for the visualization
    """
    # Identify non-zero HEIDI entries
    n = H.shape[0]
    non_zero_heidi = []
    for i in range(n):
        for j in range(n):
            if H[i, j].sum() != 0:
                non_zero_heidi.append((i, j, H[i, j].sum()))

    non_zero_heidi.sort(key=lambda x: x[2], reverse=True)

    with open(f"./results/{title}_heidi_values.txt", "w") as file:
        file.write("Points with non-zero HEIDI values:\n")
        for i, j, heidi_value in non_zero_heidi:
            file.write(f"Point {i} -> Point {j}, HEIDI Value: {heidi_value:.4f}\n")

    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the original clusters with point indices
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='Spectral', s=30, alpha=0.8)
    for idx, (x, y_val) in enumerate(X):
        ax1.text(x, y_val, str(idx), fontsize=8, ha='center', va='center')

    ax1.set_title(f"{title}\nOriginal Clusters")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    plt.colorbar(scatter, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)


    order = knn_ordering(knn_indices)

    H_combined = np.sum(H, axis=2)

    H_combined_reordered = H_combined[order][:, order]

    im = ax2.imshow(H_combined_reordered, cmap=custom_cmap, aspect='auto')
    ax2.set_title(f"{title}\nCombined HEIDI Matrix")
    ax2.axis('off')

    # Add colorbar for HEIDI matrix
    plt.colorbar(im, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)

    plt.suptitle(f"Cluster and HEIDI Matrix Visualization\n{title}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./results/{title}.png")
    plt.show()




cluster_configs = [
    {"centers": [[0, 0], [1, 1]], "std": 1.5, "title": "Highly Overlapping Clusters"},
    {"centers": [[0, 0], [4, 4]], "std": 1.0, "title": "Moderately Overlapping Clusters"},
    {"centers": [[0, 0], [10, 10]], "std": 0.1, "title": "Non-Overlapping dense Clusters"},

    {"centers": [[0, 0], [10, 10]], "std": 2.5, "title": "Non-Overlapping Clusters"}
]

# Process each cluster configuration
for config in cluster_configs:
    # Generate synthetic data
    X, y = make_blobs(
        n_samples=200, 
        centers=config["centers"], 
        cluster_std=config["std"], 
        random_state=42
    )
    
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
    
    # Visualize clusters and HEIDI matrix side by side
    visualize_clusters_and_heidi(X, y, H, P, knn_indices, config['title'])


