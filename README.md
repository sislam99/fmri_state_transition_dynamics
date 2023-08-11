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
    - Function `reproducibility(all_results, distance, WP=None, BP=None)` calculates the discrepancy between pairs of sessions in terms of the five observables of the state-transition dynamics. The inputs to this function are:
        - "all_results", which contains all the observable values for all the sessions and all the participants, calculated using `lav_observ(data, method, num_clusters)`,
        - "distance", which is either Euclidean [NM: When specifying Euclidean or cosine, no hyphenation is needed? I don't know too much about Python around here.] or cosine ( [Saiful: you need to input them as string: "Euclidean" or "cosine" ] [NM: I asked it wrong. Apology. What I wanted to ask (basic of Python) is whether the input should be with double apostrophes or not. I mean, should it be Euclidean or "Euclidean"?]); this is used for matching the states from two sessions, and
        - either within-participant discrepancy ("WP") or between-participant discrepancy ("BP") is measured. [NM: I do not understand WP=None, BP=None above. If I want to specify either of this option, what should I feed to the function? Can you be clearer? Also, WP and BP cannot be set on simultaneously, right? What is going on?] [Saiful: user needs to specify either WP= True or BP = True. If someone does not specify either of them, or does specify both as None or True then the code will raise an error.] [NM: Then, I ask you to rewrite the code (and briefly check it) such that there is only the third argument (not the third and fourth arguments) named e.g. comparison. The user should specify "WP" or "BP" as the (string) value of the "comparison" variable. If any other string is fed, it returns the error.].
    - The output of this function is a list in which each element is an numpy array consisting of the dissimilarity in the centroid position between two sessions, TV of coverage time, TV of frequency of appearance, TV of average lifespan, and the Frobenius distance between two transition probability matrices. [NM: What does each element of the list correspond to? A session pair?]

- `test_retest_reliability.py` has functions for the test-retest reliability analysis.
    - Function `ND_value(observable_results, distance)` outputs the ND value. Its inputs are:
        - "observable_results", which contains all observable values for all the sessions and all the participants [NM: Is this observable_results different from all_results? I am confused.] [Saiful: both observable_results and all_results are the same.][NM: Then, please rewrite everything (including the code using all_results instead of observable_results). This tremendously helps avoiding any confusion.] and
        - "distance"; see the explanation of `discrepancy_measures.py` above.
    - Function `permuted_ND(N, observable_results, distance)` performs the permutation test. This function outputs the ND values of the permuted results and p-values of the permutation test for five observables. 


# Sample data
We provide four dummy data files named `sample_data_participant1_session1.csv`, `sample_data_participant1_session2.csv`, `sample_data_participant2_session1.csv` and `sample_data_participant2_session2.csv`, each of which contains 8 ROIs and 1000 time points.
You first need to read these data files and convert them to numpy array before running any of the analysis above (see [example notebook](https://github.com/sislam99/fmri_state_transition_dynamics/blob/main/example.ipynb)).

# Example notebook
In the [example notebook](https://github.com/sislam99/fmri_state_transition_dynamics/blob/main/example.ipynb), we demonstrate the state-transition dynamics analysis using dummy data. We also included the test-retest reliability analysis.
