import numpy as np
import pandas as pd
from itertools import groupby
from sklearn.cluster import KMeans,  AgglomerativeClustering, BisectingKMeans
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture

def lav_observ(data, method, num_clusters):
    """
    Calculate various results and observables from the given data using a clustering method. This function can be used fonly for a single time series data. 

    Parameters:
        data (numpy array): Input data for clustering (time x number of ROIs).
        method (function): Clustering method to be used. It should take 'data' and 'num_clusters' as arguments and return cluster labels and centroids.
        num_clusters (int): Number of clusters to be generated using the clustering method.

    Returns:
        labels (numpy array): Cluster labels assigned to each data point.
        gev (float): Global Explained Variance calculated from the Global Field Power (GFP) of the data.
        results (dict): A dictionary containing other observables for the state transition dynamics, 
                       such as centroids, coverage time, frequency, average lifespan, and transition probabilities.
    """

    # Calculate the GFP from the data
    gfp = data.std(axis=1)

    # Cluster the data using the specified method
    labels, centroids = method(data, num_clusters = num_clusters)

    # Calculate observables based on the obtained cluster labels
    coverage, frequency, lifespan, trans_mat = observables4(lab=labels)

    # Store the results in a dictionary
    results = {
        "centroids": centroids,
        "coverage_time": coverage,
        "frequency": frequency,
        "average_lifespan": lifespan,
        "transition_prob": trans_mat
    }

    # Calculate the GEV using the GFP, data, and cluster labels
    gev = getting_gev(gfp, data, labels, centroids)

    # Return the cluster labels, centroids, GEV, and results dictionary
    return labels, gev, results



def observables4(lab):
    """
    Calculate various observables for the state-transition dynamics.

    Input:
        lab (numpy array or list): state labels.
    Returns:
        tuple: A tuple containing the coverage time (normalized), frequency, 
               average lifespan, and transition matrix.
    """
    
    # Counting appearance and unique appearances of microstates in the sequence
    unique_states, state_counts = np.unique(lab, return_counts=True)
    a = [k for k, v in groupby(lab)] 
    fr = np.unique(a, return_counts=True)[1]
    # Coverage time: Number of times each unique microstate appears in the sequence
    coverage_time = state_counts
    # Frequency: Proportion of time each unique microstate appears in the sequence
    freq = fr / len(lab)
    # Average lifespan: Average duration of each microstate
    average_lifespan = coverage_time / fr
    # Transition matrix: Probabilities of transitioning between microstates
    tran_mat = transition_prob(state_list=a)
    return coverage_time, freq, average_lifespan, tran_mat


def transition_prob(state_list):
    """ 
    This function calculates the transition matrix from a list of states.
    Inputs:
    state_list (list): A list containing sequential states.
    Returns:
    numpy.ndarray: A 2D numpy array representing the transition matrix.
    """
    # Convert the state_list into a DataFrame for easier manipulation
    df = pd.DataFrame(state_list)
    # Create a new column 'shift' with data shifted one space forward
    df['shift'] = df[0].shift(-1)
    # Add a count column (for group by function) to keep track of state occurrences
    df['count'] = 1
    # Group the DataFrame by the current state (0) and the next state ('shift')
    # Then, count the occurrences for each state transition and reshape the DataFrame
    trans_mat = df.groupby([0, 'shift']).count().unstack().fillna(0)
    # Normalize the transition matrix by dividing each row by the sum of row values
    # This ensures that the transition probabilities sum up to 1 for each current state
    trans_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0).values
    return trans_mat

def getting_gev(gfp, data, labels, centroids):
    """
    To calculate the Global Explained Variance (GEV) for a given dataset.
    Inputs:
        gfp (np.ndarray): 1D array representing the GFP (Global Field Power).
        data (np.ndarray): 2D array containing the data points.
        labels (np.ndarray): 1D array representing the cluster labels for each data point.
        centroids (np.ndarray): 2D array containing the centroids of the clusters.

    Returns:
        float: The calculated GEV value.
    """
    # Calculate the number of clusters based on the length of the centroids array
    n_cluster = len(centroids)

    # Calculate the Global Explained Variance (GEV) for the given dataset
    gev = sum([np.dot(cosine_similarity_1d_to_2d(centroids[k, :], data[labels == k, :]),
            (gfp[labels == k])**2) / np.sum(gfp**2) for k in range(n_cluster)])
    return gev

def cosine_similarity_1d_to_2d(y, Y):
    """
    Calculate cosine similarity between a 1D array and a 2D array.

    Parameters:
        y (numpy array): The 1D array (row vector) of length N.
        Y (numpy array): The 2D array of shape (M, N) where M is the number of rows and N is the length of each row.

    Returns:
        numpy array: An array containing the cosine similarity between y and each row of Y.
    """
    # Calculate the dot product between y and each row of Y
    dot_products = np.dot(Y, y)
    # Calculate the magnitude (Euclidean norm) of y
    magnitude_y = np.linalg.norm(y)
    # Calculate the magnitudes of each row of Y
    magnitudes_Y = np.linalg.norm(Y, axis=1)
    # Calculate cosine similarity between A and each row of B
    cosine_similarities = dot_products / (magnitude_y * magnitudes_Y)

    return cosine_similarities



################################
## METHODS#
################################
def Kmeans(data, num_clusters):
    """
    Perform K-means clustering on input data.

    Inputs:
        data (numpy array): A 2D array representing the input data (time x number of ROIs).
        num_clusters (int): Number of clusters for K-means.

    Returns:
        tuple: A tuple containing the following information:
            - labels: The cluster labels assigned to each data point in the input data.
            - centroids: Centroid of each cluster.
    """
    # Perform K-means clustering
    kmeans = KMeans(n_clusters = num_clusters,  init='k-means++', tol=0.00001, random_state=0).fit(data)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    return labels, centroids

def Kmedoids(data, num_clusters):
    """
    Perform K-medoids clustering on input data.

    Inputs:
        data (numpy array): A 2D array representing the input data (time x number of ROIs).
        num_clusters (int): Number of clusters for K-medoids.

    Returns:
        tuple: A tuple containing the following information:
            - labels: The cluster labels assigned to each data point in the input data.
            - centroids: The centroid points of the clusters.
    """
    # Perform K-medoids clustering
    kmedoids = KMedoids(n_clusters = num_clusters, init = 'k-medoids++', random_state=0).fit(data)
    labels = kmedoids.labels_
    centroids = kmedoids.cluster_centers_
 
    return labels, centroids



def AC(data, num_clusters):
    """
    Perform Agglomerative Clustering on input data.

    Inputs:
        data (numpy array): A 2D array representing the input data (time x number of ROIs).
        num_clusters (int): Number of clusters for Agglomerative Clustering.

    Returns:
        tuple: A tuple containing the following information:
            - labels: The cluster labels assigned to each data point in the input data.
            - centroids: The centroid points of the clusters.
    """
    # Perform Agglomerative Clustering
    ac = AgglomerativeClustering(n_clusters = num_clusters, linkage='ward').fit(data)
    labels = ac.labels_
    
    # Calculate centroids using mean points of each cluster
    centroids = np.array([np.mean(x, axis=0) for x in [data[labels == i] for i in range(num_clusters)]])
         
    return labels, centroids





def initial_clustering(X):
    """
    Initialize the clustering process by creating individual clusters for each data point. This initialization function only for AAHC and TAAHC.

    Parameters:
        X (numpy array): A 2D array representing the input data (time x number of ROIs).

    Returns:
        list: List of initial clusters, where each cluster is represented as a tuple containing indices of data points belonging to that cluster.
    """
    N = X.shape[0]
    # Calculate the correlation matrix among data points
    corr = np.corrcoef(X) - np.eye(N)
    
    # Find the pair of data points with the highest correlation, indicating they belong to the same initial cluster
    clust_one = (np.argmax(corr) // N, np.argmax(corr) % N)
    
    # Create a list of clusters with each data point belonging to its own cluster initially
    cluster = [clust_one]
    for k in range(N):
        if k in clust_one:
            continue
        cluster.append((k,))
    
    return cluster


def getting_labels(X, new_clusters):
    """
    Assign labels to data points based on the updated clusters.

    Parameters:
        X(numpy array): A 2D array representing the input data (time x number of ROIs).
        new_clusters (list): List of updated clusters after the AAHC/TAAHC algorithm.

    Returns:
        numpy array: Cluster labels assigned to each data point.
        numpy array: Centroids of the clusters.
    """
    N = X.shape[0]
    li = len(new_clusters)
    label = np.zeros((N, 2))
    centroids = []
    for l in range(li):
        # Assign labels to data points in the current cluster
        label[new_clusters[l], 0] = new_clusters[l]
        label[new_clusters[l], 1] = l
        # Calculate the centroid of the current cluster
        center_l = np.mean(X[new_clusters[l], :], axis=0)
        centroids.append(center_l)
    
    return label[:, 1], np.array(centroids)


def AAHC(data, num_clusters):
    """
    Perform Atomize Agglomerative  Hierarchical Clustering (AAHC) to obtain a specific number of clusters.

    Parameters:
        data (numpy array): A 2D array representing the input data (time x number of ROIs).
        num_clusters (int): Desired number of clusters.

    Returns:
        numpy array: Cluster labels assigned to each data point.
        numpy array: Centroids of the final clusters.
    """
    # Initialize clustering by creating individual clusters for each data point
    cluster = initial_clustering(X=data)

    # Calculate GFP and normalize them and use them to adaptively merge clusters
    gfp = data.std(axis=1)
    all_gfp2 = gfp**2 / sum(gfp**2)

    while len(cluster) > num_clusters:
        gev_list = []
        for k in range(len(cluster)):
            clust_data = data[cluster[k], :]
            c_size = len(cluster[k])
            clust_mean = clust_data.mean(axis=0)        
            # Calculate the correlation between the mean of the current cluster and its data points
            corr_k = correlation(clust_mean, clust_data)[0, 1:]           
            # Get the GFP values squared for the current cluster
            gfp2_k = all_gfp2[sorted(list(cluster[k]))]          
            # Calculate the Global Extremes Value (GEV) for the current cluster
            gev_k = np.dot(gfp2_k, corr_k**2)
            gev_list.append(np.mean(gev_k))
        # Find the cluster with the lowest average GEV and merge it with the most correlated cluster
        bad = cluster[np.argmin(gev_list)]
        cluster.remove(bad)
        new_cluster = cluster.copy()
        ncl = len(cluster)
        clus_cen = [np.mean(data[cluster[l], :], axis=0) for l in range(ncl)]
        for bd in bad:
            cr_list = []
            # Calculate the correlation between the current data point and the centroids of other clusters
            corr_bd = correlation(data[bd, :], clus_cen)[0, 1:]
            # Find the cluster with the highest correlation to merge the current data point
            new_idx = np.argmax(corr_bd)
            new_cluster[new_idx] = new_cluster[new_idx] + (bd,)
        cluster = new_cluster.copy()
    N = data.shape[0]
    # Get the final cluster labels and centroids
    labels, centroids = getting_labels(X=data, new_clusters=new_cluster)
    return labels, centroids



def TAAHC(data, num_clusters):
    """
    Perform the Topographic Agglomerative Algorithm with Hierarchical Clustering (TAAHC).

    Parameters:
        data (numpy array): A 2D array representing the input data (time x number of ROIs).
        num_cluster (int): Desired number of clusters.

    Returns:
        numpy array: Cluster labels assigned to each data point.
        numpy array: Centroids of the final clusters.
    """
    # Initialize clustering by creating individual clusters for each data point
    cluster = initial_clustering(X=data)
    
    # Continue merging clusters until the desired number of clusters is reached
    while len(cluster) > num_clusters:
        cor_list = []
        for k in range(len(cluster)):
            clust_data = data[cluster[k], :]
            c_size = len(cluster[k])
            clust_mean = clust_data.mean(axis=0)         
            # Calculate the correlation between the mean of the current cluster and its data points
            corr_k = correlation(clust_mean, clust_data)[0, 1:]
            avg_cor = sum(corr_k) / c_size
            cor_list.append(avg_cor)     
        # Find the cluster with the lowest average correlation and merge it with the most correlated cluster
        to_atomize = cluster[np.argmin(cor_list)]
        cluster.remove(to_atomize)
        new_cluster = cluster.copy()
        ncl = len(cluster)
        clus_cen = [np.mean(data[cluster[l], :], axis=0) for l in range(ncl)]       
        for bd in to_atomize:
            cr_list = []
            # Calculate the correlation between the current data point and the centroids of other clusters
            corr_bd = correlation(data[bd, :], clus_cen)[0, 1:]
            new_idx = np.argmax(corr_bd)
            new_cluster[new_idx] = new_cluster[new_idx] + (bd,)        
        cluster = new_cluster.copy()
    # Get the final cluster labels and centroids
    labels, centroids = getting_labels(X=data, new_clusters=new_cluster)
    return labels, centroids




def Bisecting_Kmeans(data, num_clusters):
    """
    Perform Bisecting KMeans clustering.

    Parameters:
        data (numpy array): A 2D array representing the input data (time x number of ROIs).
        num_clusters (int): Desired number of clusters.

    Returns:
        numpy array: Cluster labels assigned to each data point.
        numpy array: Centroids of the final clusters.
    """
    # Perform Bisecting KMeans clustering using the scikit-learn BisectingKMeans implementation
    bisect_means = BisectingKMeans(n_clusters=num_clusters, init='k-means++', tol=0.00001, random_state=0).fit(data)  
    # Get cluster labels and centroids from the clustering result
    labels = bisect_means.labels_
    centroids = bisect_means.cluster_centers_
    return labels, centroids



def GMM(data, num_clusters):
    """
    Perform Gaussian Mixture Model (GMM) clustering on input data.

    Parameters:
        data (numpy array): A 2D array representing the input data (time x number of ROIs).
        num_clusters (int): Number of clusters for GMM.

    Returns:
        tuple: A tuple containing the following information:
            - labels: The cluster labels assigned to each data point in the input data.
            - centroids: The centroid points of the clusters.
    """
    # Perform Gaussian Mixture Model (GMM) clustering using the scikit-learn GaussianMixture class
    gmm = GaussianMixture(n_components=num_clusters, random_state=0).fit(data)
    
    # Get cluster labels and centroids from the GMM clustering result
    labels = gmm.predict(data)
    centroids = gmm.means_

    return labels, centroids

