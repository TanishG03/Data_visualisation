

import numpy as np
import pandas as pd
from itertools import chain, combinations
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
# import squareform
from scipy.spatial.distance import squareform

def knn_ordering(X, k):
    """
    Order points using k-nearest neighbors.
    Returns ordered indices.
    """
    # Compute KNN graph
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Create adjacency matrix from KNN graph
    n = X.shape[0]
    adj_matrix = np.zeros((n, n))
    for i in range(n):
        adj_matrix[i, indices[i]] = 1
    
    # Make symmetric
    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
    
    # Use the first eigenvector for ordering
    eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
    order = np.argsort(eigenvectors[:, -2])
    
    return order

def hierarchical_ordering(X, method='ward'):
    """
    Order points using hierarchical clustering.
    Returns ordered indices.
    """
    # Compute linkage matrix
    distances = pdist(X)
    Z = linkage(distances, method=method)
    
    # Extract the order from the dendrogram
    def get_ordering(Z, n):
        if len(Z) == 0:
            return [0]
        
        ordering = []
        def recurse(node, leaf_order):
            if node < n:
                ordering.append(node)
            else:
                left = int(Z[node - n, 0])
                right = int(Z[node - n, 1])
                if leaf_order == 'left':
                    recurse(left, 'left')
                    recurse(right, 'left')
                else:
                    recurse(right, 'right')
                    recurse(left, 'right')
                    
        recurse(2*n - 2, 'left')
        return ordering
    
    order = get_ordering(Z, X.shape[0])
    return np.array(order)

def heidi_matrix_with_dual_ordering(X, D, k, n_clusters):
    """
    Compute HEIDI matrix with different ordering strategies for diagonal and off-diagonal blocks
    """
    n = X.shape[0]
    P = powerset(D)
    subspace_map = {subspace: idx for idx, subspace in enumerate(P)}
    H = np.zeros((n, n, len(P)), dtype=int)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Get cluster indices
    cluster_indices = [np.where(cluster_labels == i)[0] for i in range(n_clusters)]
    
    # Apply different ordering strategies within each cluster
    final_order = []
    cluster_boundaries = []
    current_position = 0
    
    for i, indices in enumerate(cluster_indices):
        cluster_data = X[indices]
        
        # Apply KNN ordering within cluster
        local_order = knn_ordering(cluster_data, min(k, len(indices)-1))
        ordered_indices = indices[local_order]
        
        final_order.extend(ordered_indices)
        cluster_boundaries.append((current_position, current_position + len(indices)))
        current_position += len(indices)
    
    final_order = np.array(final_order)
    
    # Compute k-nearest neighbors for each subspace
    for subspace in P:
        subspace_indices = list(subspace)
        
        # Process each cluster pair
        for i in range(n_clusters):
            for j in range(n_clusters):
                start_i, end_i = cluster_boundaries[i]
                start_j, end_j = cluster_boundaries[j]
                
                block_indices_i = final_order[start_i:end_i]
                block_indices_j = final_order[start_j:end_j]
                
                if i == j:  # Diagonal blocks - use KNN
                    block_data = X[block_indices_i][:, subspace_indices]
                    knn = NearestNeighbors(n_neighbors=min(k, len(block_indices_i)))
                    knn.fit(block_data)
                    _, nn_indices = knn.kneighbors(block_data)
                    
                    for idx, point_idx in enumerate(block_indices_i):
                        for nn_idx in nn_indices[idx]:
                            neighbor_idx = block_indices_i[nn_idx]
                            H[point_idx, neighbor_idx, subspace_map[subspace]] = 1
                            H[neighbor_idx, point_idx, subspace_map[subspace]] = 1
                
                else:  # Off-diagonal blocks - use hierarchical distances
                    block_data_i = X[block_indices_i][:, subspace_indices]
                    block_data_j = X[block_indices_j][:, subspace_indices]
                    
                    # Compute pairwise distances
                    distances = pdist(np.vstack([block_data_i, block_data_j]))
                    distance_matrix = squareform(distances)[:len(block_data_i), len(block_data_i):]
                    
                    # Connect k-nearest neighbors based on hierarchical distances
                    for idx, point_idx in enumerate(block_indices_i):
                        nn_indices = np.argsort(distance_matrix[idx])[:k]
                        for nn_idx in nn_indices:
                            neighbor_idx = block_indices_j[nn_idx]
                            H[point_idx, neighbor_idx, subspace_map[subspace]] = 1
                            H[neighbor_idx, point_idx, subspace_map[subspace]] = 1
    
    return H, P, final_order, cluster_boundaries, cluster_labels

def heidi_matrix_with_knn_strategy(X, D, k, n_clusters):
    """
    Implementation following the second approach with pure KNN strategy
    """
    n = X.shape[0]
    P = powerset(D)
    subspace_map = {subspace: idx for idx, subspace in enumerate(P)}
    H = np.zeros((n, n, len(P)), dtype=int)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Get cluster indices
    cluster_indices = [np.where(cluster_labels == i)[0] for i in range(n_clusters)]
    
    # Apply KNN ordering within each cluster
    final_order = []
    cluster_boundaries = []
    current_position = 0
    
    for indices in cluster_indices:
        cluster_data = X[indices]
        local_order = knn_ordering(cluster_data, min(k, len(indices)-1))
        ordered_indices = indices[local_order]
        
        final_order.extend(ordered_indices)
        cluster_boundaries.append((current_position, current_position + len(indices)))
        current_position += len(indices)
    
    final_order = np.array(final_order)
    
    # Compute KNN connections for each subspace
    for subspace in P:
        subspace_indices = list(subspace)
        
        # Process each cluster pair
        for i in range(n_clusters):
            for j in range(n_clusters):
                start_i, end_i = cluster_boundaries[i]
                start_j, end_j = cluster_boundaries[j]
                
                block_indices_i = final_order[start_i:end_i]
                block_indices_j = final_order[start_j:end_j]
                
                # Use KNN for both diagonal and off-diagonal blocks
                block_data_i = X[block_indices_i][:, subspace_indices]
                block_data_j = X[block_indices_j][:, subspace_indices]
                knn_i = NearestNeighbors(n_neighbors=min(k, len(block_indices_i)))
                knn_j = NearestNeighbors(n_neighbors=min(k, len(block_indices_j)))
                knn_i.fit(block_data_i)
                knn_j.fit(block_data_j)
                _, nn_indices_i = knn_i.kneighbors(block_data_i)
                _, nn_indices_j = knn_j.kneighbors(block_data_j)
                
                # Fill the matrix for both directions
                for idx, point_idx in enumerate(block_indices_i):
                    for nn_idx in nn_indices_i[idx]:
                        neighbor_idx = block_indices_i[nn_idx]
                        H[point_idx, neighbor_idx, subspace_map[subspace]] = 1
                        H[neighbor_idx, point_idx, subspace_map[subspace]] = 1
                for idx, point_idx in enumerate(block_indices_j):
                    for nn_idx in nn_indices_j[idx]:
                        neighbor_idx = block_indices_j[nn_idx]
                        H[point_idx, neighbor_idx, subspace_map[subspace]] = 1
                        H[neighbor_idx, point_idx, subspace_map[subspace]] = 1
    
    return H, P, final_order, cluster_boundaries, cluster_labels

def heidi_matrix_with_first_approach_ordering(X, D, k, n_clusters):
    """
    Implementation following the first approach's ordering style
    """
    n = X.shape[0]
    P = powerset(D)
    subspace_map = {subspace: idx for idx, subspace in enumerate(P)}
    H = np.zeros((n, n, len(P)), dtype=int)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Sort points by cluster labels first
    cluster_order = np.argsort(cluster_labels)
    ordered_labels = cluster_labels[cluster_order]
    
    # Compute cluster boundaries based on sorted labels
    cluster_boundaries = []
    current_label = ordered_labels[0]
    current_start = 0
    
    for i, label in enumerate(ordered_labels):
        if label != current_label:
            cluster_boundaries.append((current_start, i))
            current_start = i
            current_label = label
    cluster_boundaries.append((current_start, len(ordered_labels)))
    
    # For each subspace, compute KNN and populate the matrix
    for subspace in P:
        subspace_indices = list(subspace)
        knn_indices = compute_knn(X, k, subspace_indices)
        
        # Fill the matrix using cluster-ordered indices
        for i, orig_i in enumerate(cluster_order):
            for j in knn_indices[orig_i]:
                H[orig_i, j, subspace_map[subspace]] = 1
                H[j, orig_i, subspace_map[subspace]] = 1
    
    return H, P, cluster_order, cluster_boundaries, cluster_labels

def heidi_matrix_with_dual_ordering1(X, D, k, n_clusters, ordering_strategy='mix1'):
    """
    Compute HEIDI matrix with different ordering strategies for diagonal and off-diagonal blocks
    """
    n = X.shape[0]
    P = powerset(D)
    subspace_map = {subspace: idx for idx, subspace in enumerate(P)}
    H = np.zeros((n, n, len(P)), dtype=int)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Get cluster indices
    cluster_indices = [np.where(cluster_labels == i)[0] for i in range(n_clusters)]
    
    # Apply different ordering strategies within each cluster
    final_order = []
    cluster_boundaries = []
    current_position = 0
    
    for i, indices in enumerate(cluster_indices):
        cluster_data = X[indices]
        
        # Apply KNN ordering within cluster
        local_order = knn_ordering(cluster_data, min(k, len(indices)-1))
        ordered_indices = indices[local_order]
        
        final_order.extend(ordered_indices)
        cluster_boundaries.append((current_position, current_position + len(indices)))
        current_position += len(indices)
    
    final_order = np.array(final_order)
    
    # Compute k-nearest neighbors for each subspace
    for subspace in P:
        subspace_indices = list(subspace)
        
        # Process each cluster pair
        for i in range(n_clusters):
            for j in range(n_clusters):
                start_i, end_i = cluster_boundaries[i]
                start_j, end_j = cluster_boundaries[j]
                
                block_indices_i = final_order[start_i:end_i]
                block_indices_j = final_order[start_j:end_j]

                if ordering_strategy == 'mix1':
                    if i == j:  # Diagonal blocks - use KNN
                        block_data = X[block_indices_i][:, subspace_indices]
                        knn = NearestNeighbors(n_neighbors=min(k, len(block_indices_i)))
                        knn.fit(block_data)
                        _, nn_indices = knn.kneighbors(block_data)
                        
                        for idx, point_idx in enumerate(block_indices_i):
                            for nn_idx in nn_indices[idx]:
                                neighbor_idx = block_indices_i[nn_idx]
                                H[point_idx, neighbor_idx, subspace_map[subspace]] = 1
                                H[neighbor_idx, point_idx, subspace_map[subspace]] = 1
                    
                    else:  # Off-diagonal blocks - use hierarchical distances
                        block_data_i = X[block_indices_i][:, subspace_indices]
                        block_data_j = X[block_indices_j][:, subspace_indices]
                        
                        # Compute pairwise distances
                        distances = pdist(np.vstack([block_data_i, block_data_j]))
                        distance_matrix = squareform(distances)[:len(block_data_i), len(block_data_i):]
                        
                        # Connect k-nearest neighbors based on hierarchical distances
                        for idx, point_idx in enumerate(block_indices_i):
                            nn_indices = np.argsort(distance_matrix[idx])[:k]
                            for nn_idx in nn_indices:
                                neighbor_idx = block_indices_j[nn_idx]
                                H[point_idx, neighbor_idx, subspace_map[subspace]] = 1
                                H[neighbor_idx, point_idx, subspace_map[subspace]] = 1

                elif ordering_strategy == 'mix2':
                    if i == j: # Diagonal blocks - use hierarchical distances
                        block_data = X[block_indices_i][:, subspace_indices]
                        order = hierarchical_ordering(block_data)
                        ordered_indices = block_indices_i[order]
                        
                        for idx, point_idx in enumerate(ordered_indices):
                            for neighbor_idx in ordered_indices[idx+1:idx+k+1]:
                                H[point_idx, neighbor_idx, subspace_map[subspace]] = 1
                                H[neighbor_idx, point_idx, subspace_map[subspace]] = 1
                    else: # Off-diagonal blocks - use KNN
                        block_data_i = X[block_indices_i][:, subspace_indices]
                        block_data_j = X[block_indices_j][:, subspace_indices]
                        knn_i = NearestNeighbors(n_neighbors=min(k, len(block_indices_i)))
                        knn_j = NearestNeighbors(n_neighbors=min(k, len(block_indices_j)))
                        knn_i.fit(block_data_i)
                        knn_j.fit(block_data_j)
                        _, nn_indices_i = knn_i.kneighbors(block_data_i)
                        _, nn_indices_j = knn_j.kneighbors(block_data_j)
                        
                        for idx, point_idx in enumerate(block_indices_i):
                            for nn_idx in nn_indices_i[idx]:
                                neighbor_idx = block_indices_i[nn_idx]
                                H[point_idx, neighbor_idx, subspace_map[subspace]] = 1
                                H[neighbor_idx, point_idx, subspace_map[subspace]] = 1
                        for idx, point_idx in enumerate(block_indices_j):
                            for nn_idx in nn_indices_j[idx]:
                                neighbor_idx = block_indices_j[nn_idx]
                                H[point_idx, neighbor_idx, subspace_map[subspace]] = 1
                                H[neighbor_idx, point_idx, subspace_map[subspace]] = 1

                elif ordering_strategy == 'knn': # same ordering strategy for all blocks
                    if i == j:
                        block_data = X[block_indices_i][:, subspace_indices]
                        knn = NearestNeighbors(n_neighbors=min(k, len(block_indices_i)))
                        knn.fit(block_data)
                        _, nn_indices = knn.kneighbors(block_data)
                        
                        for idx, point_idx in enumerate(block_indices_i):
                            for nn_idx in nn_indices[idx]:
                                neighbor_idx = block_indices_i[nn_idx]
                                H[point_idx, neighbor_idx, subspace_map[subspace]] = 1
                                H[neighbor_idx, point_idx, subspace_map[subspace]] = 1
                    else:
                        block_data_i = X[block_indices_i][:, subspace_indices]
                        block_data_j = X[block_indices_j][:, subspace_indices]
                        knn_i = NearestNeighbors(n_neighbors=min(k, len(block_indices_i)))
                        knn_j = NearestNeighbors(n_neighbors=min(k, len(block_indices_j)))
                        knn_i.fit(block_data_i)
                        knn_j.fit(block_data_j)
                        _, nn_indices_i = knn_i.kneighbors(block_data_i)
                        _, nn_indices_j = knn_j.kneighbors(block_data_j)
                        
                        for idx, point_idx in enumerate(block_indices_i):
                            for nn_idx in nn_indices_i[idx]:
                                neighbor_idx = block_indices_i[nn_idx]
                                H[point_idx, neighbor_idx, subspace_map[subspace]] = 1
                                H[neighbor_idx, point_idx, subspace_map[subspace]] = 1
                        for idx, point_idx in enumerate(block_indices_j):
                            for nn_idx in nn_indices_j[idx]:
                                neighbor_idx = block_indices_j[nn_idx]
                                H[point_idx, neighbor_idx, subspace_map[subspace]] = 1
                                H[neighbor_idx, point_idx, subspace_map[subspace]] = 1
                    
                elif ordering_strategy == 'hierarchical':
                        if i == j:
                            block_data = X[block_indices_i][:, subspace_indices]
                            order = hierarchical_ordering(block_data)
                            ordered_indices = block_indices_i[order]
                            
                            for idx, point_idx in enumerate(ordered_indices):
                                for neighbor_idx in ordered_indices[idx+1:idx+k+1]:
                                    H[point_idx, neighbor_idx, subspace_map[subspace]] = 1
                                    H[neighbor_idx, point_idx, subspace_map[subspace]] = 1
                        else:
                            block_data_i = X[block_indices_i][:, subspace_indices]
                            block_data_j = X[block_indices_j][:, subspace_indices]
                            order_i = hierarchical_ordering(block_data_i)
                            order_j = hierarchical_ordering(block_data_j)
                            ordered_indices_i = block_indices_i[order_i]
                            ordered_indices_j = block_indices_j[order_j]
                            
                            for idx, point_idx in enumerate(ordered_indices_i):
                                for neighbor_idx in ordered_indices_j[:k]:
                                    H[point_idx, neighbor_idx, subspace_map[subspace]] = 1
                                    H[neighbor_idx, point_idx, subspace_map[subspace]] = 1

    return H, P, final_order, cluster_boundaries, cluster_labels

                    
                    


    

                
                


def powerset(s):
    """Generate all non-empty subsets of a set."""
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))

def compute_knn(X, k, subspace_indices):
    """Compute k-nearest neighbors for each point in the specified subspace."""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X[:, subspace_indices])
    distances, indices = nbrs.kneighbors(X[:, subspace_indices])
    return indices

def heidi_matrix_with_clusters(X, D, k, n_clusters):
    """
    Compute HEIDI matrix with cluster-based ordering
    
    Parameters:
    X: data matrix
    D: dimension indices
    k: number of nearest neighbors
    n_clusters: number of clusters
    """
    n = X.shape[0]
    P = powerset(D)
    subspace_map = {subspace: idx for idx, subspace in enumerate(P)}
    H = np.zeros((n, n, len(P)), dtype=int)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Sort points by cluster labels
    cluster_order = np.argsort(cluster_labels)
    ordered_labels = cluster_labels[cluster_order]
    
    # Compute cluster boundaries
    cluster_boundaries = []
    current_label = ordered_labels[0]
    current_start = 0
    
    for i, label in enumerate(ordered_labels):
        if label != current_label:
            cluster_boundaries.append((current_start, i))
            current_start = i
            current_label = label
    cluster_boundaries.append((current_start, len(ordered_labels)))
    
    # Compute k-nearest neighbors for each subspace
    for subspace in P:
        subspace_indices = list(subspace)
        knn_indices = compute_knn(X, k, subspace_indices)
        
        # Fill the HEIDI matrix using cluster-ordered indices
        for i, orig_i in enumerate(cluster_order):
            for j in knn_indices[orig_i]:
                H[orig_i, j, subspace_map[subspace]] = 1
                H[j, orig_i, subspace_map[subspace]] = 1  # Make symmetric
    
    return H, P, cluster_order, cluster_boundaries, cluster_labels

def visualize_heidi_blocks(H, P, cluster_boundaries, title="HEIDI Matrix Block Visualization"):
    """
    Visualize HEIDI matrix with cluster blocks for each subspace
    """
    # Aggregate HEIDI matrix over all subspaces
    H_combined = np.sum(H, axis=2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot the combined HEIDI matrix
    cax = ax.imshow(H_combined, cmap='Blues', aspect='auto')
    
    # Add cluster block boundaries
    for start, end in cluster_boundaries:
        # Draw horizontal lines
        ax.axhline(y=start-0.5, color='red', linewidth=1)
        # Draw vertical lines
        ax.axvline(x=start-0.5, color='red', linewidth=1)
    
    # Add block labels
    n_clusters = len(cluster_boundaries)
    for i in range(n_clusters):
        for j in range(n_clusters):
            start_i, end_i = cluster_boundaries[i]
            start_j, end_j = cluster_boundaries[j]
            center_i = (start_i + end_i) / 2
            center_j = (start_j + end_j) / 2
            
            # Add block labels
            if i == j:
                label = f"C{i+1},C{i+1}"
            else:
                label = f"C{i+1},C{j+1}"
            
            ax.text(center_j, center_i, label,
                   horizontalalignment='center',
                   verticalalignment='center',
                   color='black' if np.mean(H_combined[start_i:end_i, start_j:end_j]) < 0.5 else 'white')
    
    ax.set_title(title)
    fig.colorbar(cax)
    plt.show()


import numpy as np

def calculate_distance_weight(dist, sigma=1.0):
    """
    Calculate distance weight using Gaussian kernel
    """
    return np.exp(-dist**2 / (2 * sigma**2))

def calculate_within_cluster_affinity(ordering, affinity_matrix, positions):
    """
    Calculate within-cluster pattern affinity (WPA) for a given ordering
    
    Parameters:
    ordering (array): Array of point indices in the proposed order
    affinity_matrix (array): Matrix of affinities between points
    positions (dict): Dictionary mapping point indices to their positions
    
    Returns:
    float: Within-cluster pattern affinity score
    """
    n = len(ordering)
    wpa = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            point_i = ordering[i]
            point_j = ordering[j]
            pos_diff = abs(positions[point_i] - positions[point_j])
            weight = calculate_distance_weight(pos_diff)
            wpa += affinity_matrix[point_i, point_j] * weight
    
    return wpa

# def calculate_across_cluster_affinity(ordering, other_clusters, affinity_matrix, positions):
#     """
#     Calculate across-cluster pattern affinity (APA) for a given ordering
    
#     Parameters:
#     ordering (array): Array of point indices in the proposed order
#     other_clusters (list): List of arrays containing point indices for other clusters
#     affinity_matrix (array): Matrix of affinities between points
#     positions (dict): Dictionary mapping point indices to their positions
    
#     Returns:
#     float: Across-cluster pattern affinity score
#     """
#     apa = 0.0
    
#     for point_i in ordering:
#         for cluster in other_clusters:
#             for point_j in cluster:
#                 pos_diff = abs(positions[point_i] - positions[point_j])
#                 weight = calculate_distance_weight(pos_diff)
#                 apa += affinity_matrix[point_i, point_j] * weight
    
#     return apa

def calculate_across_cluster_affinity(ordering, other_clusters, affinity_matrix, positions):
    """
    Calculate across-cluster pattern affinity (APA) for a given ordering
    """
    apa = 0.0

    for point_i in ordering:
        for cluster in other_clusters:
            for point_j in cluster:
                # Check if point_j is in positions
                if point_j in positions:
                    pos_diff = abs(positions[point_i] - positions[point_j])
                    weight = calculate_distance_weight(pos_diff)
                    apa += affinity_matrix[point_i, point_j] * weight
                else:
                    # Handle the case where point_j is not in positions
                    # For instance, you could assign a default weight or skip
                    apa += affinity_matrix[point_i, point_j]  # No weight
                    
    return apa

def calculate_block_affinities(H, cluster_boundaries, final_order):
    """
    Calculate within and across cluster affinities for each block in the HEIDI matrix
    
    Parameters:
    H (array): HEIDI matrix (n x n x |P|)
    cluster_boundaries (list): List of (start, end) tuples for each cluster
    final_order (array): The ordering of points
    
    Returns:
    dict: Dictionary containing affinity scores for each block
    """
    n_clusters = len(cluster_boundaries)
    # Aggregate HEIDI matrix over all subspaces to get affinity matrix
    affinity_matrix = np.sum(H, axis=2)
    
    block_scores = {}
    
    for i in range(n_clusters):
        start_i, end_i = cluster_boundaries[i]
        block_indices_i = final_order[start_i:end_i]
        
        # Calculate within-cluster affinity for diagonal blocks
        positions = {idx: pos for pos, idx in enumerate(block_indices_i)}
        
        # For diagonal blocks (within-cluster)
        within_affinity = calculate_within_cluster_affinity(
            block_indices_i, 
            affinity_matrix,
            positions
        )
        
        # For off-diagonal blocks (across-cluster)
        across_affinities = []
        other_clusters = []
        for j in range(n_clusters):
            if i != j:
                start_j, end_j = cluster_boundaries[j]
                block_indices_j = final_order[start_j:end_j]
                other_clusters.append(block_indices_j)
        
        across_affinity = calculate_across_cluster_affinity(
            block_indices_i,
            other_clusters,
            affinity_matrix,
            positions
        )
        
        block_scores[f"C{i+1}"] = {
            'within_affinity': within_affinity,
            'across_affinity': across_affinity,
            'quality_score': within_affinity - across_affinity,
            'ordering': block_indices_i.tolist()
        }
    
    return block_scores

# Function to print results in a formatted way
def print_block_affinity_results(block_scores):
    """Print the affinity scores for each block in a readable format"""
    print("\nBlock Affinity Scores:")
    print("-" * 50)
    for block, scores in block_scores.items():
        print(f"\nBlock {block}:")
        print(f"Ordering: {scores['ordering']}")
        print(f"Within-cluster affinity: {scores['within_affinity']:.4f}")
        print(f"Across-cluster affinity: {scores['across_affinity']:.4f}")
        print(f"Quality score: {scores['quality_score']:.4f}")

# Example usage
import os
# def main(data, n_clusters=3, k=10):
def main(filepath):


    _, file_extension = os.path.splitext(filepath)
    
    if os.path.basename(filepath) == 'Iris.csv':
        data = pd.read_csv(filepath).iloc[:, 1:-1]
    elif file_extension.lower() == '.csv':
        data = pd.read_csv(filepath).iloc[:, :-1]
    elif file_extension.lower() in ('.xls', '.xlsx'):
        data = pd.read_excel(filepath).iloc[:, :-1]
    else:
        raise ValueError("Unsupported file format")

    # Parameters
    n_clusters = 3
    k = 10
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Set up dimension indices
    D = range(scaled_data.shape[1])
    
    # Compute HEIDI matrix with dual ordering strategy
    # H, P, final_order, cluster_boundaries, cluster_labels = heidi_matrix_with_dual_ordering(
    #     scaled_data, D, k, n_clusters
    # )
    
    # # Visualize the block structure
    # visualize_heidi_blocks(H, P, cluster_boundaries)

    # # After computing H, P, final_order, cluster_boundaries, cluster_labels:
    # block_scores = calculate_block_affinities(H, cluster_boundaries, final_order)
    # print_block_affinity_results(block_scores)

    Hmix1, P, final_order, cluster_boundaries, cluster_labels = heidi_matrix_with_dual_ordering1(
        scaled_data, D, k, n_clusters, 'mix1'
    )

    Hmix2, P, final_order, cluster_boundaries, cluster_labels = heidi_matrix_with_dual_ordering1(
        scaled_data, D, k, n_clusters, 'mix2'
    )

    Hknn, P, final_order, cluster_boundaries, cluster_labels = heidi_matrix_with_dual_ordering1(
        scaled_data, D, k, n_clusters, 'knn'
    )

    Hhierarchical, P, final_order, cluster_boundaries, cluster_labels = heidi_matrix_with_dual_ordering1(
        scaled_data, D, k, n_clusters, 'hierarchical'
    )

    visualize_heidi_blocks(Hmix1, P, cluster_boundaries, title="HEIDI Matrix Block Visualization - Mix1")
    visualize_heidi_blocks(Hmix2, P, cluster_boundaries, title="HEIDI Matrix Block Visualization - Mix2")
    visualize_heidi_blocks(Hknn, P, cluster_boundaries, title="HEIDI Matrix Block Visualization - KNN")
    visualize_heidi_blocks(Hhierarchical, P, cluster_boundaries, title="HEIDI Matrix Block Visualization - Hierarchical")

    block_scores_mix1 = calculate_block_affinities(Hmix1, cluster_boundaries, final_order)
    block_scores_mix2 = calculate_block_affinities(Hmix2, cluster_boundaries, final_order)
    block_scores_knn = calculate_block_affinities(Hknn, cluster_boundaries, final_order)
    block_scores_hierarchical = calculate_block_affinities(Hhierarchical, cluster_boundaries, final_order)

    print_block_affinity_results(block_scores_mix1)
    print_block_affinity_results(block_scores_mix2)
    print_block_affinity_results(block_scores_knn)
    print_block_affinity_results(block_scores_hierarchical)

    Hknn2, P, final_order, cluster_boundaries, cluster_labels = heidi_matrix_with_knn_strategy(
        scaled_data, D, k, n_clusters
    )

    Hknn1, P, final_order, cluster_boundaries, cluster_labels = heidi_matrix_with_first_approach_ordering(
        scaled_data, D, k, n_clusters
    )

    visualize_heidi_blocks(Hknn2, P, cluster_boundaries, title="HEIDI Matrix Block Visualization - KNN Strategy")
    visualize_heidi_blocks(Hknn1, P, cluster_boundaries, title="HEIDI Matrix Block Visualization - First Approach Ordering")

    block_scores_knn2 = calculate_block_affinities(Hknn2, cluster_boundaries, final_order)
    block_scores_knn1 = calculate_block_affinities(Hknn1, cluster_boundaries, final_order)

    print_block_affinity_results(block_scores_knn2)
    print_block_affinity_results(block_scores_knn1)
    


    
    # return H, cluster_labels


# Example usage
# if __name__ == "__main__":
#     # Load the dataset
#     data = pd.read_csv('Iris.csv').iloc[:, 1:-1]
    
#     # Process the data and visualize HEIDI matrix
#     main(data)