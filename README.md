# State-transition dynamics of resting state fMRI data

When you use the code provided here, please cite the following paper:
*citation will be here*

This repository provides the code written in Python for the following two analyses:
- Compute dynamics of discrete states of multi-variate resting state fMRI data determined with one of the seven clustering methods.
- Assess the test-retest reliability of the estimated dynamics of discrete states. 

# How to use the code?
We divided the entire analysis into a few parts and for each part provided a Python (.py) file containing relevant functions and required libraries. 

- `all_observables.py` contains a few functions. These functions can be used to determine the 
cluster labels for each time point, centroids of the clusters, and calculate the GEV and other observables of the estimated state-transition dynamics (coverage time, frequency of each state, average lifespan, transition probability matrix).
    - For example, function `Kmeans(data, num_clusters)` takes the time series data (i.e., "data") and the number of clusters (i.e., "num_clusters") as input, and outputs the cluster label of each time point and the centroid of each cluster.
    - Function `lav_observ(data, method, num_clusters)` outputs the cluster label of each time point, GEV, and the five observables (centroid position, coverage time, frequency of each state, average lifespan, transition probability matrix). Its inputs are:
        - the time series data,
        - clustering method (Kmeans, Kmedoids, AC, AAHC, TAAHC, Bisecting_Kmeans, or GMM), and
        - the number of clusters.

- `discrepancy_measures.py` has functions which altogether provide discrepancy measures of observables between two sessions of the same participant and between two sessions of different participants.
    - Function `reproducibility(all_results, distance, comparison)` calculates the discrepancy between pairs of sessions in terms of the five observables of the state-transition dynamics. The inputs to this function are:
        - "all_results", which contains all the observable values for all the sessions and all the participants, calculated using `lav_observ(data, method, num_clusters)`,
        - "distance", which should be either 'Euclidean' or 'cosine'. This is used for matching the states from two sessions, and
        - "comparison", which should be either 'WP' (for within-participant comparison) or 'BP' (for between-participant comparison). 
    - The output of this function is a list in which each element is an numpy array consisting of the dissimilarity in the centroid position between two sessions, TV of coverage time between the same two sessions, TV of frequency of each state between the two sessions, TV of average lifespan between the two sessions, and the Frobenius distance between the two transition probability matrices. The length of the list of the number of session pairs to be considered.

- `test_retest_reliability.py` has functions for the test-retest reliability analysis.
    - Function `ND_value(all_results, distance)` outputs the ND value. Its inputs are:
        - "all_results", which contains all observable values for all the sessions and all the participants, and
        - "distance"; see the explanation of `discrepancy_measures.py` above.
    - Function `permuted_ND(N, all_results, distance)` performs the permutation test. This function outputs the ND values of the permuted results and p-values of the permutation test for five observables. 


# Sample data
We provide four dummy data files named `sample_data_participant1_session1.csv`, `sample_data_participant1_session2.csv`, `sample_data_participant2_session1.csv` and `sample_data_participant2_session2.csv`, each of which contains 8 ROIs and 1000 time points.
You first need to read these data files and convert them to numpy array before running any of the analysis above (see [example notebook](https://github.com/sislam99/fmri_state_transition_dynamics/blob/main/example.ipynb)).

# Example notebook
In the [example notebook](https://github.com/sislam99/fmri_state_transition_dynamics/blob/main/example.ipynb), we demonstrate the state-transition dynamics analysis using dummy data. We also included the test-retest reliability analysis.
