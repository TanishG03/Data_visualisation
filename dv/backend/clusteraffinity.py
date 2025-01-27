import numpy as np
from scipy.spatial.distance import pdist, squareform

def calculate_distance_weight(dist, sigma=1.0):
    """
    Calculate distance weight using Gaussian kernel
    
    Parameters:
    dist (float): Distance between points
    sigma (float): Width parameter for Gaussian kernel
    
    Returns:
    float: Weight value between 0 and 1
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

def calculate_across_cluster_affinity(ordering, other_clusters, affinity_matrix, positions):
    """
    Calculate across-cluster pattern affinity (APA) for a given ordering
    
    Parameters:
    ordering (array): Array of point indices in the proposed order
    other_clusters (list): List of arrays containing point indices for other clusters
    affinity_matrix (array): Matrix of affinities between points
    positions (dict): Dictionary mapping point indices to their positions
    
    Returns:
    float: Across-cluster pattern affinity score
    """
    apa = 0.0
    
    for point_i in ordering:
        for cluster in other_clusters:
            for point_j in cluster:
                pos_diff = abs(positions[point_i] - positions[point_j])
                weight = calculate_distance_weight(pos_diff)
                apa += affinity_matrix[point_i, point_j] * weight
    
    return apa

def evaluate_ordering_quality(ordering, other_clusters, affinity_matrix, alpha=1.0, beta=1.0):
    """
    Evaluate the quality of a cluster ordering considering both within and across cluster affinities
    
    Parameters:
    ordering (array): Array of point indices in the proposed order
    other_clusters (list): List of arrays containing point indices for other clusters
    affinity_matrix (array): Matrix of affinities between points
    alpha (float): Weight for within-cluster affinity
    beta (float): Weight for across-cluster affinity
    
    Returns:
    float: Quality score (higher is better)
    """
    # Create position mapping
    positions = {idx: pos for pos, idx in enumerate(ordering)}
    for cluster in other_clusters:
        for pos, idx in enumerate(cluster):
            positions[idx] = pos
    
    # Calculate within-cluster affinity
    wpa = calculate_within_cluster_affinity(ordering, affinity_matrix, positions)
    
    # Calculate across-cluster affinity
    apa = calculate_across_cluster_affinity(ordering, other_clusters, affinity_matrix, positions)
    
    # Calculate final quality score
    quality = alpha * wpa - beta * apa
    
    return quality

def compare_orderings(o1, o2, other_clusters, affinity_matrix, alpha=1.0, beta=1.0):
    """
    Compare two orderings to determine which is better
    
    Parameters:
    o1 (array): First ordering to compare
    o2 (array): Second ordering to compare
    other_clusters (list): List of arrays containing point indices for other clusters
    affinity_matrix (array): Matrix of affinities between points
    alpha (float): Weight for within-cluster affinity
    beta (float): Weight for across-cluster affinity
    
    Returns:
    bool: True if o1 is better than o2, False otherwise
    float: Quality difference between o1 and o2
    """
    q1 = evaluate_ordering_quality(o1, other_clusters, affinity_matrix, alpha, beta)
    q2 = evaluate_ordering_quality(o2, other_clusters, affinity_matrix, alpha, beta)
    
    return q1 > q2, q1 - q2

# Example usage
def demonstrate_ordering_evaluation():
    """
    Demonstrate how to use the ordering evaluation functions
    """
    # Create sample affinity matrix
    # n_points = 10
    # affinity_matrix = np.random.rand(n_points, n_points)
    # affinity_matrix = (affinity_matrix + affinity_matrix.T) / 2  # Make symmetric
    # np.fill_diagonal(affinity_matrix, 1.0)

    # # print affinity_matrix
    # print(affinity_matrix)
    
    # # Create sample clusters
    # c1 = np.array([0, 1, 2, 3])
    # c2 = np.array([4, 5, 6])
    # c3 = np.array([7, 8, 9])
    
    # # Create two different orderings for c1
    # o1 = np.array([0, 1, 2, 3])  # Original order
    # o2 = np.array([3, 1, 0, 2])  # Alternative order
    
    # other_clusters = [c2, c3]
    
    # # Compare orderings
    # is_o1_better, quality_diff = compare_orderings(o1, o2, other_clusters, affinity_matrix)
    affinity_matrix = np.array([
    [1.0, 0.8, 0.4, 0.3, 0.2, 0.1],
    [0.8, 1.0, 0.5, 0.3, 0.1, 0.2],
    [0.4, 0.5, 1.0, 0.7, 0.1, 0.3],
    [0.3, 0.3, 0.7, 1.0, 0.2, 0.4],
    [0.2, 0.1, 0.1, 0.2, 1.0, 0.9],
    [0.1, 0.2, 0.3, 0.4, 0.9, 1.0]
])

# Sample clusters
    c1 = np.array([0, 1, 2, 3])
    c2 = np.array([4, 5])

    # Two orderings to compare within c1
    o1 = np.array([0, 1, 2, 3])
    o2 = np.array([3, 1, 0, 2])

    # Compare the two orderings
    is_ordering_1_better, quality_diff = compare_orderings(o1, o2, [c2], affinity_matrix)
    
    print(f"Ordering 1: {o1}")
    print(f"Ordering 2: {o2}")
    print(f"Is ordering 1 better? {o1}")
    print(f"Quality difference: {quality_diff}")
    
    return o1, quality_diff

if __name__ == "__main__":
    demonstrate_ordering_evaluation()