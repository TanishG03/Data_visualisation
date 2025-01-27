import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from itertools import chain, combinations
import matplotlib.colors as mcolors

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

def generate_dense_concentric_spheres(n_samples, noise=0.02, n_shells=3, base_radius=1, shell_spacing=0.5):
    """
    Generate multiple concentric spherical shells with controlled density and spacing.
    
    Parameters:
    - n_samples: Total number of points
    - noise: Noise level for shell perturbation
    - n_shells: Number of concentric shells
    - base_radius: Radius of innermost shell
    - shell_spacing: Distance between consecutive shells
    """
    points_per_shell = n_samples // n_shells
    
    X_list = []
    y_list = []
    
    for shell in range(n_shells):
        # Generate uniform points on sphere using Fibonacci spiral
        radius = base_radius + shell * shell_spacing
        indices = np.arange(0, points_per_shell, dtype=float) + 0.5
        
        phi = np.arccos(1 - 2*indices/points_per_shell)
        theta = np.pi * (1 + 5**0.5) * indices
        
        # Convert to Cartesian coordinates
        x = radius * np.cos(theta) * np.sin(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(phi)
        
        # Add controlled noise
        noise_vector = np.random.normal(0, noise, (points_per_shell, 3))
        shell_points = np.column_stack((x, y, z)) + noise_vector
        
        X_list.append(shell_points)
        y_list.append(np.full(points_per_shell, shell))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y

def generate_dense_torus(n_samples, noise=0.02, R=3, r=1, n_major=50, n_minor=20):
    """
    Generate a dense torus with controlled point distribution.
    
    Parameters:
    - n_samples: Total number of points
    - noise: Noise level
    - R: Major radius
    - r: Minor radius
    - n_major: Number of points along major circumference
    - n_minor: Number of points along minor circumference
    """
    # Create a grid of angles for even distribution
    theta = np.linspace(0, 2*np.pi, n_major)
    phi = np.linspace(0, 2*np.pi, n_minor)
    theta, phi = np.meshgrid(theta, phi)
    
    # Flatten the angles and repeat to reach desired n_samples
    theta = theta.flatten()
    phi = phi.flatten()
    
    # Repeat points if necessary to reach n_samples
    n_points = len(theta)
    repetitions = n_samples // n_points + 1
    theta = np.tile(theta, repetitions)[:n_samples]
    phi = np.tile(phi, repetitions)[:n_samples]
    
    # Add small random perturbation to angles for more natural distribution
    theta += np.random.normal(0, 0.1/n_major, n_samples)
    phi += np.random.normal(0, 0.1/n_minor, n_samples)
    
    # Generate coordinates
    x = (R + r*np.cos(phi)) * np.cos(theta)
    y = (R + r*np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    
    # Add controlled noise
    noise_vector = np.random.normal(0, noise, (n_samples, 3))
    X = np.column_stack((x, y, z)) + noise_vector
    y = np.zeros(n_samples)
    
    return X, y

def generate_dense_3d_spiral(n_samples, noise=0.05, n_turns=5, height=10, radius_growth=0.5):
    """
    Generate a dense 3D spiral with controlled geometry.
    
    Parameters:
    - n_samples: Number of points
    - noise: Noise level
    - n_turns: Number of complete turns
    - height: Total height of spiral
    - radius_growth: Rate of radius increase per turn
    """
    # Generate more uniform point distribution along spiral
    t = np.linspace(0, n_turns*2*np.pi, n_samples)
    
    # Calculate radius with controlled growth
    radius = 1 + radius_growth * t/(2*np.pi)
    
    # Generate coordinates
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = height * t/(n_turns*2*np.pi)
    
    # Add controlled noise
    noise_vector = np.random.normal(0, noise, (n_samples, 3))
    X = np.column_stack((x, y, z)) + noise_vector
    y = np.zeros(n_samples)
    
    return X, y

def generate_interleaved_helices(n_samples, noise=0.02, n_helices=2, radius=2, pitch=1, n_turns=3):
    """
    Generate multiple interleaved helical structures.
    
    Parameters:
    - n_samples: Total number of points
    - noise: Noise level
    - n_helices: Number of interleaved helices
    - radius: Base radius of helices
    - pitch: Vertical distance per turn
    - n_turns: Number of complete turns
    """
    points_per_helix = n_samples // n_helices
    
    X_list = []
    y_list = []
    
    for i in range(n_helices):
        t = np.linspace(0, n_turns*2*np.pi, points_per_helix)
        phase = 2*np.pi*i/n_helices  # Phase shift for each helix
        
        x = radius * np.cos(t + phase)
        y = radius * np.sin(t + phase)
        z = pitch * t/(2*np.pi)
        
        # Add controlled noise
        noise_vector = np.random.normal(0, noise, (points_per_helix, 3))
        helix_points = np.column_stack((x, y, z)) + noise_vector
        
        X_list.append(helix_points)
        y_list.append(np.full(points_per_helix, i))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y

# Update the dataset generation function to use the new structures
def generate_3d_datasets1():
    """Generate diverse 3D clustering datasets with enhanced structure definition."""
    datasets = [
        {
            "name": "dense_concentric_spheres",
            "generator": generate_dense_concentric_spheres,
            "params": {
                "n_samples": 150,
                "noise": 0.02,
                "n_shells": 3,
                "base_radius": 1,
                "shell_spacing": 0.7
            },
            "title": "Dense Concentric Spheres"
        },
        {
            "name": "dense_torus",
            "generator": generate_dense_torus,
            "params": {
                "n_samples": 150,
                "noise": 0.02,
                "R": 3,
                "r": 1,
                "n_major": 50,
                "n_minor": 20
            },
            "title": "Dense Toroidal Structure"
        },
        {
            "name": "dense_spiral",
            "generator": generate_dense_3d_spiral,
            "params": {
                "n_samples": 150,
                "noise": 0.05,
                "n_turns": 5,
                "height": 10,
                "radius_growth": 0.5
            },
            "title": "Dense 3D Spiral"
        },
        {
            "name": "interleaved_helices",
            "generator": generate_interleaved_helices,
            "params": {
                "n_samples": 150,
                "noise": 0.02,
                "n_helices": 3,
                "radius": 2,
                "pitch": 1,
                "n_turns": 3
            },
            "title": "Interleaved Helical Structures"
        }
    ]
    return datasets

def generate_3d_datasets():
    """Generate diverse 3D clustering datasets for HEIDI analysis."""
    datasets = [
        # Concentric Spherical Shells
        {
            "name": "concentric_spheres",
            "generator": lambda n_samples, noise: generate_concentric_spheres(n_samples, noise),
            "params": {
                "n_samples": 500,
                "noise": 0.05
            },
            "title": "Concentric Spherical Shells"
        },
        # Interleaved Spirals
        {
            "name": "spiral_3d",
            "generator": lambda n_samples, noise: generate_3d_spiral(n_samples, noise),
            "params": {
                "n_samples": 500,
                "noise": 0.1
            },
            "title": "3D Spiral Structure"
        },
        # Overlapping Gaussians
        {
            "name": "overlapping_gaussians",
            "generator": make_blobs,
            "params": {
                "n_samples": 500,
                "centers": [[0, 0, 0], [1, 1, 1]],
                "cluster_std": [1.0, 0.5],
                "random_state": 42,
                "n_features": 3
            },
            "title": "Overlapping 3D Gaussians"
        },
        # Toroidal Structure
        {
            "name": "torus",
            "generator": lambda n_samples, noise: generate_3d_torus(n_samples, noise),
            "params": {
                "n_samples": 500,
                "noise": 0.05
            },
            "title": "3D Toroidal Structure"
        }
    ]
    return datasets

def generate_concentric_spheres(n_samples, noise=0.05):
    """Generate concentric spherical shells in 3D."""
    n_per_sphere = n_samples // 2
    
    # Generate angles
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    phi = np.random.uniform(0, np.pi, n_samples)
    
    # Inner sphere (radius = 1)
    r1 = 1 + np.random.normal(0, noise, n_per_sphere)
    x1 = r1 * np.sin(phi[:n_per_sphere]) * np.cos(theta[:n_per_sphere])
    y1 = r1 * np.sin(phi[:n_per_sphere]) * np.sin(theta[:n_per_sphere])
    z1 = r1 * np.cos(phi[:n_per_sphere])
    
    # Outer sphere (radius = 2)
    r2 = 2 + np.random.normal(0, noise, n_per_sphere)
    x2 = r2 * np.sin(phi[n_per_sphere:]) * np.cos(theta[n_per_sphere:])
    y2 = r2 * np.sin(phi[n_per_sphere:]) * np.sin(theta[n_per_sphere:])
    z2 = r2 * np.cos(phi[n_per_sphere:])
    
    X = np.vstack([np.column_stack((x1, y1, z1)), np.column_stack((x2, y2, z2))])
    y = np.concatenate([np.zeros(n_per_sphere), np.ones(n_per_sphere)])
    
    return X, y

def generate_3d_spiral(n_samples, noise=0.1):
    """Generate a 3D spiral structure."""
    t = np.linspace(0, 10*np.pi, n_samples)
    x = t * np.cos(t) + np.random.normal(0, noise, n_samples)
    y = t * np.sin(t) + np.random.normal(0, noise, n_samples)
    z = t + np.random.normal(0, noise, n_samples)
    
    X = np.column_stack((x, y, z))
    y = np.zeros(n_samples)  # Single cluster
    
    return X, y

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

def generate_3d_torus(n_samples, noise=0.05):
    """Generate a 3D torus structure."""
    R = 3  # major radius
    r = 1  # minor radius
    
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    phi = np.random.uniform(0, 2*np.pi, n_samples)
    
    x = (R + r*np.cos(phi)) * np.cos(theta)
    y = (R + r*np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    
    # Add noise
    x += np.random.normal(0, noise, n_samples)
    y += np.random.normal(0, noise, n_samples)
    z += np.random.normal(0, noise, n_samples)
    
    X = np.column_stack((x, y, z))
    y = np.zeros(n_samples)  # Single cluster
    
    return X, y

def visualize_3d_clusters_and_heidi(X, y, H, P, knn_indices, title):
    """Create a visualization of 3D clusters and corresponding HEIDI matrix with both ordered and unordered views."""
    fig = plt.figure(figsize=(20, 10))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='Spectral', s=30, alpha=0.6)
    ax1.set_title(f"{title}\n3D Clusters")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    
    # Get the kNN ordering
    order = knn_ordering(knn_indices)
    
    # Aggregate HEIDI matrices
    H_combined = np.sum(H, axis=2)
    
    # Plot unordered HEIDI matrix
    ax2 = fig.add_subplot(132)
    im1 = ax2.imshow(H_combined, cmap='viridis', aspect='auto')
    ax2.set_title(f"Unordered HEIDI Matrix\n{title}")
    plt.colorbar(im1, ax=ax2)
    
    # Plot ordered HEIDI matrix
    H_combined_reordered = H_combined[order][:, order]
    ax3 = fig.add_subplot(133)
    im2 = ax3.imshow(H_combined_reordered, cmap='viridis', aspect='auto')
    ax3.set_title(f"Ordered HEIDI Matrix\n{title}")
    plt.colorbar(im2, ax=ax3)
    
    # Add text annotation showing the ordering
    fig.text(0.02, 0.02, f"KNN Ordering first 10 indices: {order[:10]}...", fontsize=8, wrap=True)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"./results/3d_{title.replace(' ', '_')}.png")
    plt.close()

def export_visualization_data(X, H, knn_indices, title):
    """Export data for visualization with both ordered and unordered matrices."""
    # Get the KNN ordering
    order = knn_ordering(knn_indices)
    
    # Aggregate HEIDI matrices
    H_combined = np.sum(H, axis=2)
    H_combined_reordered = H_combined[order][:, order]
    
    # Convert numpy types to native Python types
    visualization_data = {
        'points': [
            {'x': float(x), 'y': float(y), 'z': float(z)} 
            for x, y, z in X
        ],
        'heidi_matrix': {
            'unordered': [
                [float(val) for val in row]
                for row in H_combined
            ],
            'ordered': [
                [float(val) for val in row]
                for row in H_combined_reordered
            ]
        },
        'ordering': [int(i) for i in order],
        'knn_indices': [[int(i) for i in row] for row in knn_indices]
    }
    
    import json
    with open(f'./visualization_data_final_{title.replace(" ", "_")}.json', 'w') as f:
        json.dump(visualization_data, f)

def save_heidi_values(H, knn_indices, title):
    """Save both ordered and unordered non-zero HEIDI values to a file."""
    H_combined = np.sum(H, axis=2)
    order = knn_ordering(knn_indices)
    H_combined_reordered = H_combined[order][:, order]
    
    with open(f"./results/3d_{title}_heidi_values.txt", "w") as file:
        # Save unordered values
        file.write(f"Unordered non-zero HEIDI values for {title}:\n")
        for i in range(H_combined.shape[0]):
            for j in range(H_combined.shape[1]):
                if H_combined[i, j] != 0:
                    file.write(f"Point {i} -> Point {j}: {H_combined[i,j]:.4f}\n")
        
        file.write("\n" + "="*50 + "\n\n")
        
        # Save ordered values
        file.write(f"Ordered non-zero HEIDI values for {title}:\n")
        for i in range(H_combined_reordered.shape[0]):
            for j in range(H_combined_reordered.shape[1]):
                if H_combined_reordered[i, j] != 0:
                    file.write(f"Point {order[i]} -> Point {order[j]}: {H_combined_reordered[i,j]:.4f}\n")

def run_3d_heidi_analysis():
    """Run HEIDI analysis on 3D datasets."""
    datasets = generate_3d_datasets1()
    
    for dataset_config in datasets:
        # Generate data
        if dataset_config["name"] in ["concentric_spheres", "spiral_3d", "torus"]:
            X, y = dataset_config["generator"](**dataset_config["params"])
        else:
            X, y = dataset_config["generator"](**dataset_config["params"])
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        
        # Define analysis parameters
        D = range(scaled_X.shape[1])
        k = 5  # Increased k for 3D space
        
        # Compute k-nearest neighbors
        knn_indices = compute_knn(scaled_X, k, list(D))
        
        # Compute HEIDI matrix
        H, P = heidi_matrix(scaled_X, D, k)
        
        # Visualize results
        visualize_3d_clusters_and_heidi(X, y, H, P, knn_indices, dataset_config['title'])
        export_visualization_data(X, H, knn_indices, dataset_config['title'])
        
        # Save HEIDI values
        save_heidi_values(H, knn_indices, dataset_config['title'])

# def visualize_3d_clusters_and_heidi(X, y, H, P, knn_indices, title):
#     """Create a visualization of 3D clusters and corresponding HEIDI matrix."""
#     fig = plt.figure(figsize=(15, 5))
    
#     # 3D scatter plot
#     ax1 = fig.add_subplot(121, projection='3d')
#     scatter = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='Spectral', s=30, alpha=0.6)
#     ax1.set_title(f"{title}\n3D Clusters")
#     ax1.set_xlabel("X")
#     ax1.set_ylabel("Y")
#     ax1.set_zlabel("Z")
    
#     # Get the kNN ordering
#     order = knn_ordering(knn_indices)
    
#     # Aggregate and plot HEIDI matrix
#     H_combined = np.sum(H, axis=2)
#     H_combined_reordered = H_combined[order][:, order]
    
#     ax2 = fig.add_subplot(122)
#     im = ax2.imshow(H_combined_reordered, cmap='viridis', aspect='auto')
#     ax2.set_title(f"HEIDI Matrix\n{title}")
#     plt.colorbar(im, ax=ax2)
    
#     plt.tight_layout()
#     plt.show()
#     plt.savefig(f"./results/3d_{title.replace(' ', '_')}.png")
#     plt.close()

# def export_visualization_data(X, H, knn_indices, title):
#     """Export data for visualization with only essential information."""
#     # Get the KNN ordering
#     order = knn_ordering(knn_indices)
    
#     # Aggregate HEIDI matrix
#     H_combined = np.sum(H, axis=2)
#     H_combined_reordered = H_combined[order][:, order]
    
#     # Convert numpy types to native Python types
#     visualization_data = {
#         'points': [
#             {'x': float(x), 'y': float(y), 'z': float(z)} 
#             for x, y, z in X
#         ],
#         'heidi_matrix': [
#             [float(val) for val in row]
#             for row in H_combined_reordered
#         ]
#     }
    
#     import json
#     with open(f'./visualization_data_{title.replace(" ", "_")}.json', 'w') as f:
#         json.dump(visualization_data, f)

# def run_3d_heidi_analysis():
#     """Run HEIDI analysis on 3D datasets."""
#     datasets = generate_3d_datasets1()
    
    
#     for dataset_config in datasets:
#         # Generate data
#         if dataset_config["name"] in ["concentric_spheres", "spiral_3d", "torus"]:
#             X, y = dataset_config["generator"](**dataset_config["params"])
#         else:
#             X, y = dataset_config["generator"](**dataset_config["params"])
        
#         # Standardize the data
#         scaler = StandardScaler()
#         scaled_X = scaler.fit_transform(X)
        
#         # Define analysis parameters
#         D = range(scaled_X.shape[1])
#         k = 15  # Increased k for 3D space
        
#         # Compute k-nearest neighbors
#         knn_indices = compute_knn(scaled_X, k, list(D))
        
#         # Compute HEIDI matrix
#         H, P = heidi_matrix(scaled_X, D, k)
        
#         # Visualize results
#         visualize_3d_clusters_and_heidi(X, y, H, P, knn_indices, dataset_config['title'])
#         export_visualization_data(X, H, knn_indices, dataset_config['title'])
        
#         # Save HEIDI values
#         save_heidi_values(H, dataset_config['title'])

# def save_heidi_values(H, title):
#     """Save non-zero HEIDI values to a file."""
#     H_combined = np.sum(H, axis=2)
#     non_zero_heidi = []
    
#     for i in range(H_combined.shape[0]):
#         for j in range(H_combined.shape[1]):
#             if H_combined[i, j] != 0:
#                 non_zero_heidi.append((i, j, H_combined[i, j]))
    
#     non_zero_heidi.sort(key=lambda x: x[2], reverse=True)
    
#     with open(f"./results/3d_{title}_heidi_values.txt", "w") as file:
#         file.write(f"Non-zero HEIDI values for {title}:\n")
#         for i, j, value in non_zero_heidi:
#             file.write(f"Point {i} -> Point {j}: {value:.4f}\n")

if __name__ == "__main__":
    run_3d_heidi_analysis()