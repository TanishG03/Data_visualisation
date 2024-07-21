import pandas as pd
from scipy.spatial import distance
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

def getDistances(allData, centroid):
    row, col = allData.shape
    distances = []
    for i in range(row):
        dist = distance.euclidean(allData.iloc[i, :-1], centroid)
        distances.append(dist)
    allData['distance'] = distances
    allData = allData.sort_values(['distance'], ascending=False)
    return allData

def closest_node(allData, node):
    row, col = allData.shape
    dist = np.sqrt(np.sum((allData.iloc[:, :-3].values - node)**2, axis=1))
    index = np.argmin(dist[~allData['done']])
    return index    

def getConnectedDistances(trainingSet, point):
    row, col = trainingSet.shape    
    trainingSet['done'] = False
    trainingSet['pos'] = -1
    count = 0
    while not all(trainingSet['done']):
        rownum = closest_node(trainingSet.copy(), point)
        trainingSet.at[rownum, 'done'] = True
        trainingSet.at[rownum, 'pos'] = count
        count += 1
        point = trainingSet.loc[rownum, :-1].copy()
    trainingSet = trainingSet.sort_values(['pos'], ascending=False)
    return trainingSet.iloc[:, :-1]
 
def minSpanningTree(trainingSet, testInstance):
    row, col = trainingSet.shape    
    dists = distance.cdist(trainingSet.iloc[:, :-1].values, trainingSet.iloc[:, :-1].values, 'euclidean')
    mst = minimum_spanning_tree(dists)
    trainingSet['done'] = False
    trainingSet['pos'] = -1    
    count = 0
    mst = mst.toarray().astype(float)
    rownum = closest_node(trainingSet, testInstance)
    point = trainingSet.loc[rownum, :-1].copy()
    trainingSet.at[rownum, 'done'] = True
    trainingSet.at[rownum, 'pos'] = count
    stack = [trainingSet.index.get_loc(rownum)]
    count += 1
    while len(stack) > 0:
        rownum = stack.pop()
        rownum_orig = trainingSet.index.get_values()[rownum]
        point = trainingSet.loc[rownum_orig, :-1].copy()
        temp = list(np.nonzero(mst[rownum]))[0]
        temp1 = list(np.nonzero(mst[:, rownum]))[0]
        temp = list(temp) + list(temp1)    
        for i in temp:
            k = trainingSet.index.get_values()[i]
            if trainingSet.at[k, 'done'] == False:
                stack.extend([i])
                trainingSet.at[k, 'done'] = True
                trainingSet.at[k, 'pos'] = count
                count += 1

    trainingSet = trainingSet.sort_values(['pos'], ascending=False)
    return trainingSet.iloc[:, :-1]

def sortbasedOnclassLabel(feature_vector, ordermeasure):
    centroids = pd.DataFrame()
    for k in set(feature_vector.iloc[:, -1]):
        x = feature_vector[feature_vector.iloc[:, -1] == k].mean().to_frame().T
        centroids = pd.concat([centroids, x], ignore_index=True)
    
    sorted_data = []
    for i in set(feature_vector.iloc[:, -1]):
        if ordermeasure == 'centroid_distance':
            temp = getDistances(feature_vector[feature_vector.iloc[:, -1] == i].copy(), centroids[centroids.iloc[:, -1] == i].iloc[:, :-1].values[0])
            sorted_data.append(temp)
            
        elif ordermeasure == 'connected_distance': 
            temp = getConnectedDistances(feature_vector[feature_vector.iloc[:, -1] == i].copy(), centroids.loc[i, :].iloc[:-1].values)
            sorted_data.append(temp)
            
        elif ordermeasure == 'mst_distance':
            temp = minSpanningTree(feature_vector[feature_vector.iloc[:, -1] == i].copy(), centroids.loc[i, :].iloc[:-1].values)
            sorted_data.append(temp)
    
    sorted_data = pd.concat(sorted_data, ignore_index=True)
    return sorted_data

def main(feature_vector):
    return sortbasedOnclassLabel(feature_vector, 'centroid_distance')
