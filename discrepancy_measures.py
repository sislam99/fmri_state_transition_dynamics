import numpy as np
import pandas as pd
import itertools
from itertools import groupby, combinations

def reproducibility(all_results, distance, WP=None, BP=None):
    """
    Calculate reproducibility measures between multiple sessions or subjects.

    Parameters:
        all_results (numpy array): A 2D array containing results such as centroids, coverage time, frequency, average lifespan, and transition probability matrix of multiple sessions or subjects.
        distance (str or None): The distance measure to be used for matching centroids. Options are "cosine", "Euclidean", or None.
        WP (bool or None): If True, the analysis will be performed within subjects. If False or None, it will be performed within subjects (subjects).
        BP (bool or None): If True, the analysis will be performed between subjects. If False or None, it will be performed between subjects (subjects).

    Returns:
        list: A list containing reproducibility measures between all pairs of sessions or subjects.
            - The list contains arrays for each pair, with elements corresponding to:
                - Element 0: Dissimilarity between matched centroids.
                - Element 1: Temporal variation of coverage time between matched clusters.
                - Element 2: Temporal variation of frequency between matched clusters.
                - Element 3: Temporal variation of average lifespan between matched clusters.
                - Element 4: Frobenius norm of the difference in transition probability matrices between matched clusters.
    """
    P, S = all_results.shape
    if (WP == True) & (BP == None):
        # If within-subject analysis, consider all pairs of sessions
        all_pairs = np.array(list(combinations(list(range(S)), 2)))
        R = P
    if (WP == None) & (BP == True):
        # If between-subject analysis, consider all pairs of subjects
        all_pairs = np.array(list(combinations(list(range(P)), 2)))
        R = S
        all_results = all_results.T

    across_measurements = []
    
    for r in range(R):
        for pr in all_pairs:
            inf1, inf2 = all_results[r, pr]

            # Match centroids using the specified distance measure
            if len(inf1["centroids"]) < 9:
                # If the number of centroids is small, use the optimal matching approach
                matched1, matched2, match_dist = matching_pair(centroid1=inf1["centroids"],
                                                               centroid2=inf2["centroids"],distance=distance)
            else:
                # If the number of centroids is large, use the greedy matching approach
                matched1, matched2, match_dist = greedy_matching_pair(centroid1=inf1["centroids"],
                                                                      centroid2=inf2["centroids"],distance=distance)
            
            # Calculate the temporal variations of coverage time, frequency, and average lifespan between matched clusters
            tv_cov = max(abs(inf1["coverage_time"][matched1] - inf2["coverage_time"][matched2]))
            tv_freq = max(abs(inf1["frequency"][matched1] - inf2["frequency"][matched2]))
            tv_life = max(abs(inf1["average_life_span"][matched1] - inf2["average_life_span"][matched2]))

            # Calculate the Frobenius norm of the difference in transition probability matrices between matched clusters
            arange_trans1 = (inf1["transition_prob"][matched1, :])[:, matched1]
            arange_trans2 = (inf2["transition_prob"][matched2, :])[:, matched2]
            frob_tran = np.sqrt(np.sum(np.abs(arange_trans1 - arange_trans2) ** 2))

            # Store the reproducibility measures for this pair
            across_measurements.append(np.array([match_dist, tv_cov, tv_freq, tv_life, frob_tran])) 
    
    return across_measurements


        
def matching_pair(centroid1, centroid2, distance):
    """
    Find the optimal matching pairs between two sets of centroids using a permutation approach.
    Parameters:
        centroid1 (numpy array): Centroid points of the first set.
        centroid2 (numpy array): Centroid points of the second set.
        distance (str or None): The distance measure to be used for dissimilarity calculation.
                                Options are "cosine", "Euclidean", or None.

    Returns:
        list: Indices of the matched centroids from the first set.
        list: Indices of the matched centroids from the second set.
        float: The mean dissimilarity value between the matched centroids.
    """
    K = len(centroid1)
    combination = [(range(K), x) for x in itertools.permutations(range(K), len(range(K)))]
    dissim_mat = dissimilarity(centroid1, centroid2, distance)
    all_values = np.array([sum(abs(dissim_mat[combination[ci]])) / K for ci in range(len(combination))])
    min_idx = np.argmin(all_values)
    min_value = all_values[min_idx]
    return list(combination[min_idx][0]), list(combination[min_idx][1]), min_value



def greedy_matching_pair(centroid1, centroid2, distance):
    """
    Find matching pairs between two sets of centroids using a greedy approach.

    Parameters:
        centroid1 (numpy array): Centroid points of the first set.
        centroid2 (numpy array): Centroid points of the second set.
        distance (str or None): The distance measure to be used for dissimilarity calculation.
                                Options are "cosine", "Euclidean", or None.

    Returns:
        list: Indices of the matched centroids from the first set.
        list: Indices of the matched centroids from the second set.
        float: The mean dissimilarity value between the matched centroids.
    """
    dissim_mat = dissimilarity(centroid1, centroid2, distance)
    n_rows, n_cols = dissim_mat.shape
    pairs = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    pairs.sort(key=lambda x: dissim_mat[x[0], x[1]])
    row_set, col_set = set(), set()
    matched1, matched2 = [], []
    for i, j in pairs:
        if i not in row_set and j not in col_set:
            matched1.append(i)
            matched2.append(j)
            row_set.add(i)
            col_set.add(j)
    return matched1, matched2, np.mean(dissim_mat[matched1, matched2])

def dissimilarity(A, B, distance=None):
    """
    Calculate the dissimilarity between two numpy arrays A and B based on the specified distance measure.

    Parameters:
        A (numpy array): The first input array.
        B (numpy array): The second input array.
        distance (str or None): The distance measure to be used. Options are "cosine", "Euclidean", or None.
                                If None, "cosine" distance is used.

    Returns:
        numpy array: The dissimilarity matrix between A and B.
    """
    if (distance == None) | (distance == "cosine"):
        similarity = np.inner(A, B) / np.outer((A * A).sum(axis=1) ** 0.5, (B * B).sum(axis=1) ** 0.5)
        dissim = 1 - similarity
    elif distance == "Euclidean":
        # Reshape the arrays to have a new axis
        A_reshaped = A[:, np.newaxis, :]
        B_reshaped = B[np.newaxis, :, :]
        # Calculate Euclidean distances
        dissim = np.linalg.norm(A_reshaped - B_reshaped, axis=2)
    else:
        raise ValueError("Invalid distance measure. Use 'cosine', 'Euclidean', or None.")
    return dissim
