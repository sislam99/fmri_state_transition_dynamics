# State-transition dynamics of resting state fMRI data

When you use the code provided here, please cite the following paper:
*citation will be here**

# About this repository
This repository provides the code written in Python for doing the following two analyses:
- Compute dynamics of discrete states of multi-variate resting state fMRI data determined with one of the seven clustering methods.
- Assess the test-retest reliability of the estimated dynamics of discrete states. 

# How to use the code?
As our paper describes, there are a few steps from determining clusters to the test-retest reliability test. We divided the whole analysis into a few parts and for each part provided the Python (.py) file with helping functions and required libraries. 

- `all_observables.py` contains a few functions. These functions can be used to determine the 
Cluster labels, centroids, GEV, and other observables (coverage time, frequency, average lifespan, transition probability matrix).
    - For example, function `Kmeans(data, num_clusters)` takes the time series data (i.e., "data") and the number of clusters (i.e., "num_clusters") as input, and outputs the cluster label of each time point and the centroid of each cluster.
    - Function `lav_observ(data, method, num_clusters)` outputs the cluster label of each time point, GEV, and all five observables((centroid position, coverage time, frequency, average lifespan, transition probability matrix). Its inputs are the time series data, clustering method (Kmeans, Kmedoids, AC, AAHC, TAAHC, Bisecting_Kmeans, or GMM), and the number of clusters.


- `discrepancy_measures.py` has functions which altogether provide discrepancy measures of observables between sessions of the same participant and between different participants.
    - Function `reproducibility(all_results, distance, WP=None, BP=None)` will receive all observables results "all_results" of multiple sessions from multiple participants, parameter "distance" (either Euclidean or cosine) which is used for matching the states from two sessions, and either within participant ("WP") discrepancy or between participants ("BP"). The output of this function is a list in which each element is an numpy array consisting the dissimilarity between centroid position, TV of coverage time, frequency, average lifespan and the Frobenius distance between two transition probability matrices.

- `test_retest_reliability.py` has functions for test-retest reliability test.
    - Function `ND_value(observable_results, distance)` provides the ND value by taking observables of multiple sessions from multiple participants.
    - Function `permuted_ND(N, observable_results, distance)` performs the permutation test. It produces the ND values of the permuted results and p-values of the permutation test for five observables. 


# Sample data
We provide four dummy data files named `sample_data_participant1_session1.csv`, `sample_data_participant1_session2.csv`, `sample_data_participant2_session1.csv` and `sample_data_participant2_session2.csv` each of which contains 8 ROIs and 1000 time points.
You first nee to read these data files and convert them to numpy array before running any of the analysis above (see [example notebook](https://github.com/sislam99/fmri_state_transition_dynamics/blob/main/example.ipynb)).

# Example notebook
In the [example notebook](https://github.com/sislam99/fmri_state_transition_dynamics/blob/main/example.ipynb), we demonstrate the state-transition dynamics analysis using dummy data. We also included the test-retest reliability analysis.
