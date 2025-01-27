import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from itertools import chain, combinations
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

import numpy as np

def generate_datasets():
    """Generate diverse single clustering datasets with different shapes for HEIDI analysis."""
    datasets = [
        # Linear Shape
        {
            "data": generate_linear_cluster(n_samples=300),
            "title": "Linear Shape Cluster"
        },
        # Perfect Spherical Shape
        {
            "data": generate_spherical_cluster(n_samples=300, radius=0.5),
            "title": "Perfect Spherical Shape Cluster"
        },
        # Scattered Spherical Shape
        {
            "data": generate_spherical_cluster(n_samples=300, radius=1.0, scattered=True),
            "title": "Scattered Spherical Shape Cluster"
        },
        # Oval Shape
        {
            "data": generate_oval_cluster(n_samples=300, axis_lengths=(10, 1)),
            "title": "Oval Shape Cluster"
        },
        # Concentric Shape
        {
            "data": generate_concentric_clusters(n_samples=300, inner_radius=0.5, outer_radius=1.0),
            "title": "Concentric Shape Cluster"
        }
    ]
    return datasets

def generate_linear_cluster(n_samples):
    x = np.linspace(-5, 5, n_samples)
    y = np.zeros(n_samples)
    return np.column_stack((x, y))

def generate_spherical_cluster(n_samples, radius=1.0, scattered=False):
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    if scattered:
        radii = np.random.uniform(0, radius, n_samples)
    else:
        radii = np.full(n_samples, radius)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def generate_oval_cluster(n_samples, axis_lengths):
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    radii = np.random.uniform(0, 1, n_samples)
    x = axis_lengths[0] * radii * np.cos(angles)
    y = axis_lengths[1] * radii * np.sin(angles)
    return np.column_stack((x, y))

def generate_concentric_clusters(n_samples, inner_radius, outer_radius):
    half_samples = n_samples // 2
    inner_angles = np.random.uniform(0, 2 * np.pi, half_samples)
    outer_angles = np.random.uniform(0, 2 * np.pi, n_samples - half_samples)
    
    inner_x = inner_radius * np.cos(inner_angles)
    inner_y = inner_radius * np.sin(inner_angles)
    outer_x = outer_radius * np.cos(outer_angles)
    outer_y = outer_radius * np.sin(outer_angles)
    
    inner_cluster = np.column_stack((inner_x, inner_y))
    outer_cluster = np.column_stack((outer_x, outer_y))
    return np.vstack((inner_cluster, outer_cluster))


def compute_knn(X, k, subspace_indices):
    """Compute k-nearest neighbors for each point in the specified subspace."""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X[:, subspace_indices])
    distances, indices = nbrs.kneighbors(X[:, subspace_indices])
    return indices

def powerset(s):
    """Generate all non-empty subsets of a set."""
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))

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


from matplotlib.colors import LinearSegmentedColormap


def create_custom_colormap():
    """Create a custom colormap to emphasize value ranges."""
    colors = ["#000000", "#FF4500", "#FFFF00", "#32CD32", "#0000FF"]  # Black, Red, Yellow, Green, Blue
    n_bins = [0.0, 0.25, 0.5, 0.75, 1.0]  # Define intervals for the color spectrum
    cmap_name = "contrast_cmap"
    return LinearSegmentedColormap.from_list(cmap_name, list(zip(n_bins, colors)))

def visualize_heidi_analysis(X, y, H, P, knn_indices, title):
    """
    Enhanced visualization of clusters and HEIDI matrix with additional analysis.
    
    Parameters:
    - X: Original data points
    - y: Cluster labels
    - H: HEIDI matrix
    - P: Subspace combinations
    - knn_indices: k-nearest neighbors indices
    - title: Title for the visualization
    """
    # Compute silhouette score
    try:
        sil_score = silhouette_score(X, y)
    except:
        sil_score = None

    # Set up the figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Original Clusters
    scatter = axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='Spectral', s=30, alpha=0.8)
    axes[0].set_title(f"{title}\nOriginal Clusters")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    plt.colorbar(scatter, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
    
    # Add silhouette score to the title if computed
    if sil_score is not None:
        axes[0].set_title(f"{title}\nSilhouette Score: {sil_score:.4f}")

    # Aggregate the HEIDI matrix over all subspaces
    H_combined = np.sum(H, axis=2)

    # Plot 2: Combined HEIDI Matrix (Raw)
    im1 = axes[1].imshow(H_combined, cmap='viridis', aspect='auto')
    axes[1].set_title("Raw HEIDI Matrix")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

    # Plot 3: Normalized HEIDI Matrix
    H_normalized = H_combined / np.max(H_combined)
    im2 = axes[2].imshow(H_normalized, cmap='plasma', aspect='auto')
    axes[2].set_title("Normalized HEIDI Matrix")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)

    plt.suptitle(f"HEIDI Matrix Analysis\n{title}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./results/heidi_analysis_{title.replace(' ', '_')}.png")
    plt.close()

    # Generate detailed report
    report = {
        "title": title,
        "silhouette_score": sil_score,
        "heidi_matrix_stats": {
            "max_value": np.max(H_combined),
            "mean_value": np.mean(H_combined),
            "median_value": np.median(H_combined)
        }
    }
    
    return report

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

    # Sort the non_zero_heidi list based on the final HEIDI values
    non_zero_heidi.sort(key=lambda x: x[2], reverse=True)

    # Print non-zero HEIDI values
    # print("Points with non-zero HEIDI values:")
    # for i, j, heidi_value in non_zero_heidi:
    #     print(f"Point {i} -> Point {j}, HEIDI Value: {heidi_value:.4f}")

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

    # Get the kNN ordering
    order = knn_ordering(knn_indices)

    # Aggregate the HEIDI matrix over all subspaces to get a single combined matrix
    H_combined = np.sum(H, axis=2)

    # Reorder the combined matrix based on kNN ordering
    H_combined_reordered = H_combined[order][:, order]

    # Plot the HEIDI matrix with the custom colormap
    im = ax2.imshow(H_combined_reordered, cmap=custom_cmap, aspect='auto')
    ax2.set_title(f"{title}\nCombined HEIDI Matrix")
    ax2.axis('off')

    # Add colorbar for HEIDI matrix
    plt.colorbar(im, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)

    plt.suptitle(f"Cluster and HEIDI Matrix Visualization\n{title}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./results/{title.replace(' ', '_')}_heidi_analysis.png")
    plt.show()



def run_heidi_analysis():
    """Run comprehensive HEIDI analysis on multiple dataset configurations."""
    analysis_reports = []
    
    # Generate and analyze diverse datasets
    datasets = generate_datasets()
    
    for dataset_config in datasets:
        data = dataset_config["data"]
        title = dataset_config["title"]
        X = StandardScaler().fit_transform(data)
        y = np.zeros(X.shape[0])
        D = list(range(X.shape[1]))
        k = 5
        
        # Compute the HEIDI matrix
        H, P = heidi_matrix(X, D, k)
        
        # Compute k-nearest
        knn_indices = compute_knn(X, k, D)
        visualize_clusters_and_heidi(X, y, H, P, knn_indices, title)
       

# Execute the analysis
if __name__ == "__main__":
    run_heidi_analysis()