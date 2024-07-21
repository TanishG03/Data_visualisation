import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sort as sort
from sklearn.neighbors import NearestNeighbors
import math

def read_dataset(filepath):
    _, file_extension = os.path.splitext(filepath)
    
    if os.path.basename(filepath) == 'Iris.csv':
        # Assuming first column is ID and last column is class label
        data = pd.read_csv(filepath).iloc[:, 1:-1]
    elif file_extension.lower() == '.csv':
        data = pd.read_csv(filepath)
    elif file_extension.lower() in ('.xls', '.xlsx'):
        data = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format")

    # Separate feature vector and class labels
    feature_vector = data.iloc[:, :-1]  # Exclude the last column (class label)
    classLabel_given = data.iloc[:, -1]  # Only the last column (class label)
    class_label_dict = None  # You may need to define this based on your dataset
    row, col = data.shape[0], data.shape[1] - 1  # Number of rows and columns
    
    return feature_vector, classLabel_given, class_label_dict, row, col

def generateHeidiMatrix(sorted_data, row, col, nofKNN=10):
    k = nofKNN  # 1-NN
    heidi_matrix = np.zeros(shape=(row, row), dtype=np.uint64)
    max_count = int(math.pow(2, col))
    allsubspaces = range(1, max_count)
    f = lambda a: sorted(a, key=lambda x: sum(int(d) for d in bin(x)[2:]))
    allsubspaces = f(allsubspaces)
    frmt = str(col) + 'b'
    factor = 1
    bit_subspace = {}
    count = 0
    for i in allsubspaces:
        bin_value = str(format(i, frmt))
        bin_value = bin_value[::-1]
        subspace_col = [index for index, value in enumerate(bin_value) if value == '1']
        bit_subspace[count] = subspace_col
        count = count + 1
        subspace = sorted_data.iloc[:, subspace_col]
        np_subspace = subspace.values
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(np_subspace)
        temp = nbrs.kneighbors_graph(np_subspace).toarray()
        temp = temp.astype(np.uint64)
        heidi_matrix = heidi_matrix + temp * factor
        factor = factor * 2
    return heidi_matrix, bit_subspace

def visualizeHeidiImage(heidi_matrix, bit_subspace, row):
    max_count = len(bit_subspace)
    r = int(max_count / 3)
    g = int(max_count / 3)
    b = max_count - r - g
    
    x = heidi_matrix >> (max_count - r)
    y = (heidi_matrix & ((pow(2, g) - 1) << b)) >> b
    z = (heidi_matrix & (pow(2, b) - 1))

    heidi_img = np.dstack((x, y, z))
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    ax1.imshow(heidi_img, interpolation='nearest', picker=True)
    plt.savefig('nba_new.png')
    plt.show()

    cluster_matrix = []
    for i in range(0, row):
        for j in range(0, row):
            cluster_matrix.append([i, j, x[i][j], y[i][j], z[i][j]])
            
    cluster_matrix = np.array(cluster_matrix)
    return cluster_matrix, heidi_img


def main(filepath):
    # Load your data using read_dataset function
    feature_vector, _, class_label_dict, row, col = read_dataset(filepath)
    
    # Sort data based on some criteria
    sorted_data = sort.main(feature_vector)
    
    # Generate Heidi matrix
    heidi_matrix, bit_subspace = generateHeidiMatrix(sorted_data, row, col, nofKNN=10)
    
    # Visualize Heidi matrix
    cluster_matrix, heidi_img = visualizeHeidiImage(heidi_matrix, bit_subspace, row)
    
    # Save Heidi matrix to CSV (if needed)
    np.savetxt("nba_new.csv", heidi_matrix, delimiter=",")
    
    # Example of returning data and visualization (adjust as needed)
    return {"sorted_data": sorted_data, "heidi_matrix": heidi_matrix}, {"cluster_matrix": cluster_matrix, "heidi_img": heidi_img}

if __name__ == "__main__":
    main('Iris.csv')
