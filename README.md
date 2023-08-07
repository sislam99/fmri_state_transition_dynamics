# State-transition dynamics of resting state fMRI data.

When you use the code provided here, please cite the following paper:
*citation will be here**

# About this repository.
This repository provides the code written in Python for computing dynamics of discrete states of multi-variate resting state fMRI data determined with multiple clustering methods, and the test-retest reliability of the dynamics of discrete states. 

# How to use?
According to the article, there are few steps from determining clusters to reliability test. We divided the whole analysis in few parts and for each parts we provided the Python (.py) file with helping functions and required libraries. 

- `all_observables.py` file contains few functions. These functions can be used to determine the 
Cluster labels, centroids, GEV and other observables (coverage time, frequency, average lifespan, transition probability matrix). For example, `Kmeans(data, num_clusters)` will take time series data: `data` and number of cluster:`num_clusters` as input, and will provide labels of each time points and centroid of each cluster. The function `lav_observ(data, method, num_clusters)` is written to provide labels of each time points, GEV, and all five observables((centroid position, coverage time, frequency, average lifespan, transition probability matrix) for a given time series data, method (Kmeans, Kmedoids, AC, AAHC, TAAHC, Bisecting_Kmeans, GMM).


- `discrepancy_measures.py`, this consists of functions which altogether provide discrepancy measures of observables between sessions of the same participant and between different participant. The function `reproducibility(all_results, distance, WP=None, BP=None)` will receive all observables results `all_results` of multiple sessions from multiple participants, parameter "distance" (either Euclidean or cosine) which will be used for matching states from two sessions, and either within participant ("WP") discrepancy or between participants ("BP"). The output of this function is a list where each element is an numpy array consisting the dissimilarity between centroid posistion, TV of coverage time, frequency, average lifespan and the Frobenius distance between two transition matrices.

- `test_retest_reliability.py` file has several function for test-retest reliability test. The function `ND_value(observable_results, distance)` provides ND value by taking observables of multiple sessions from multiple participants. `permuted_ND(N, observable_results, distance)` function perform the permutation test. It produces the ND values of the permuted results and p-values of the permutation test for five observables. 


# Required software and packages:

# Sample dataset:
We provide four binarized dummy data files named `SampleData_Binarized_Participant_i_Session_j.mat`, each of which contains 7 ROIs and 1000 time points.  $i=1,2$ denotes the participants, and $j=1,2$ denotes the sessions.
If you run `Energy_landscape_analysis.m`, it reads these .mat files and run the analysis. If you want to run the analysis with different data sets, you only need to replace the data file name in line 19 and 56 in `Energy_landscape_analysis.m`.

# Example in notebook

