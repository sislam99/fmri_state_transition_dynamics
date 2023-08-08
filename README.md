# State-transition dynamics of resting state fMRI data

When you use the code provided here, please cite the following paper:
*citation will be here**

# About this repository
This repository provides the code written in Python for doing the following two analyses:
- Compute dynamics of discrete states of multi-variate resting state fMRI data determined with one of the seven clustering methods.
- Assess the test-retest reliability of the estimated dynamics of discrete states. 

# How to use?
According to the article, there are few steps from determining clusters to reliability test. We divided the whole analysis in few parts and for each parts we provided the Python (.py) file with helping functions and required libraries. 

- `all_observables.py` contains a few functions. These functions can be used to determine the 
Cluster labels, centroids, GEV and other observables (coverage time, frequency, average lifespan, transition probability matrix). For example, `Kmeans(data, num_clusters)` will take time series data: "data" and number of cluster: "num_clusters" as input, and will provide labels of each time points and centroid of each cluster. The function `lav_observ(data, method, num_clusters)` is written to provide labels of each time points, GEV, and all five observables((centroid position, coverage time, frequency, average lifespan, transition probability matrix) for a given time series data, method (Kmeans, Kmedoids, AC, AAHC, TAAHC, Bisecting_Kmeans, GMM).


- `discrepancy_measures.py` consists of functions which altogether provide discrepancy measures of observables between sessions of the same participant and between different participants. The function `reproducibility(all_results, distance, WP=None, BP=None)` will receive all observables results "all_results" of multiple sessions from multiple participants, parameter "distance" (either Euclidean or cosine) which will be used for matching states from two sessions, and either within participant ("WP") discrepancy or between participants ("BP"). The output of this function is a list where each element is an numpy array consisting the dissimilarity between centroid position, TV of coverage time, frequency, average lifespan and the Frobenius distance between two transition matrices.

- `test_retest_reliability.py` has functions for test-retest reliability test. The function `ND_value(observable_results, distance)` provides the ND value by taking observables of multiple sessions from multiple participants. `permuted_ND(N, observable_results, distance)` function performs the permutation test. It produces the ND values of the permuted results and p-values of the permutation test for five observables. 


# Sample dataset
We provide four dummy data files named `sample_data_participant1_session1.csv`, `sample_data_participant1_session2.csv`, `sample_data_participant2_session1.csv` and `sample_data_participant2_session2.csv` each of which contains 8 ROIs and 1000 time points.
First need to read these data files and convert them as numpy array and then you can do analyze the state-dynamics (see [example notebook](https://github.com/sislam99/fmri_state_transition_dynamics/blob/main/example.ipynb) ). To conduct the analysis with different data sets, one needs to replace the data files. 

# Example notebook
In the [example notebook](https://github.com/sislam99/fmri_state_transition_dynamics/blob/main/example.ipynb), we demonstrated the state-transition dynamics analysis using dummy data. We also showed the the test-retest reliability. 

