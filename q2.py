import numpy as np
import matplotlib.pyplot as plt

def loadDataset():
    '''
    Implemented for you.
    loads 150 examples of 2D data from kmeansdata.npz
    Returns:
    data: 2x150 numpy array where each column is a 2D data point
    '''
    data = np.load('kmeansdata.npz')
    data = np.stack((data['x'], data['y']), axis=0)
    return data

def initializeClusters(data, k):
    '''
    Implemented for you.
    Initializes k clusters by randomly selecting k data points from the dataset
    Args:
    data: 2x150 numpy array where each column is a 2D data point
    k: number of clusters
    Returns:
    clusters: 2xk numpy array where each column is a 2D data point
    '''
    clusters = data[:, np.random.choice(data.shape[1], k, replace=False)]
    return clusters

def assignClusters(data, clusters):
    '''
    Implement this.
    Assigns each data point to the nearest cluster
    Args:
    data: 2x150 numpy array where each column is a 2D data point
    clusters: 2xk numpy array where each column is a 2D data point
    Returns:
    assignments: 1x150 numpy array where each element is the index of the assigned cluster
    '''
    # YOUR CODE HERE
    assignments = None
    # END YOUR CODE
    return assignments

def updateClusters(data, assignments, k):
    '''
    Implement this.
    Updates the cluster centers by taking the mean of all the data points assigned to it
    Args:
    data: 2x150 numpy array where each column is a 2D data point
    assignments: 1x150 numpy array where each element is the index of the assigned cluster
    k: number of clusters
    Returns:
    clusters: 2xk numpy array where each column is a 2D data point
    '''
    # YOUR CODE HERE
    clusters = None
    # END YOUR CODE
    return clusters

def costKMeans(data, clusters, assignments):
    '''
    Implemented for you. 
    Computes the cost function of the k-means algorithm
    Args:
    data: 2x150 numpy array where each column is a 2D data point
    clusters: 2xk numpy array where each column is a 2D data point
    assignments: 1x150 numpy array where each element is the index of the assigned cluster
    Returns:
    cost: cost of the k-means algorithm
    '''
    return np.sum(np.linalg.norm(data - clusters[:, assignments], axis=0) ** 2)

def trainKMeans(data, k, maxIter):
    '''
    Implemented for you. 
    Trains the k-means algorithm on the given dataset
    Args:
    data: 2x150 numpy array where each column is a 2D data point
    k: number of clusters
    maxIter: maximum number of iterations
    Returns:
    clusters: 2xk numpy array where each column is a 2D data point
    assignments: 1x150 numpy array where each element is the index of the assigned cluster
    '''
    clusters = initializeClusters(data, k)
    minCost = np.inf
    for i in range(k*k):
        for i in range(maxIter):
            assignments = assignClusters(data, clusters)
            new_clusters = updateClusters(data, assignments, k)
            if np.all(clusters == new_clusters):
                break
            clusters = new_clusters
        cost = costKMeans(data, clusters, assignments)
        if cost < minCost:
            minCost = cost
            best_clusters = clusters
            best_assignments = assignments
    return best_clusters, best_assignments, minCost

def visualizeResult(data, clusters, assignments, minCost):
    '''
    Implemented for you. 
    Visualizes the data and the clusters
    Args:
    data: 2x150 numpy array where each column is a 2D data point
    clusters: 2xk numpy array where each column is a 2D data point
    assignments: 1x150 numpy array where each element is the index of the assigned cluster
    '''
    colors = plt.cm.rainbow(np.linspace(0, 1, clusters.shape[1]))
    for i in range(clusters.shape[1]):
        plt.scatter(data[0, assignments == i], data[1, assignments == i], c=colors[i], label='Cluster ' + str(i))
        plt.scatter(clusters[0, i], clusters[1, i], c='black', marker='x')
    plt.title('K-means clustering with k = ' + str(clusters.shape[1]) + ', cost = ' + str(minCost))
    plt.legend()
    plt.show()

