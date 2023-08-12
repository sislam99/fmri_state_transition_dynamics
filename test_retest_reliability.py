import numpy as np
from multiprocessing import Pool, cpu_count
from discrepancy_measures import *

def permuted_ND(N, all_results, distance):
    """
    Calculate the permuted ND values by shuffling the observable_results N times.

    Parameters:
        N (int): Number of permutations to be performed.
        all_results: (This contains observables for multiple subjects or sessions.
        distance (str or None): The distance measure to be used for centroid matching. Options are "cosine", "Euclidean", or None.

    Returns:
        numpy array: An array with permuted ND values for each permutation.
            - The shape of the array is (N, 5), where 5 corresponds to the number of observables.
    """
    # Create a list of parameters for each worker
    params_list = [(N, all_results, distance)] * cpu_count()

    # Initialize the pool of workers
    with Pool(processes = cpu_count()) as pool:
        # Distribute the computations to workers and collect the results
        results = pool.map(permuted_ND_worker, params_list)

    # Concatenate the results from all workers into a single array
    ND_perm = np.concatenate(results)
    
    # Empirical ND value
    ND_empirical = ND_value(all_results = all_results, distance = distance)
    
    # Calculate p-values for each observables 
    p_value = np.sum(ND_perm>ND_empirical, axis = 0)/N
    
    return ND_empirical, ND_perm, p_value

def ND_value(all_results, distance):
    """
    Calculate the ND (Normalized Distance) value between within-subject and between-subject  observables.

    Parameters:
        all_results: This contains observables for multiple subjects or sessions.
        distance (str or None): The distance measure to be used for centroid matching. Options are "cosine", "Euclidean", or None.

    Returns:
        numpy array: The ND value calculated as the mean of between-subject reproducibility divided by the mean of within-subject reproducibility.
    """
    within_sub = reproducibility(all_results = all_results, distance = distance, comparison = "WP")
    between_sub = reproducibility(all_results = all_results, distance = distance, comparison = "BP")
    ND = np.mean(between_sub, axis=0) / np.mean(within_sub, axis=0)
    return ND

def random_permutation(data):
    """
    Generate a random permutation of the input data.

    Parameters:
        data (numpy array): Input data to be shuffled.

    Returns:
        numpy array: The shuffled data with the same shape as the input data.
    """
    # Convert the input data to a NumPy array
    data = np.array(data)

    data_shape = data.shape
    flat_data = data.flatten()
    np.random.shuffle(flat_data)
    shuffled_data = flat_data.reshape(data_shape)
    return shuffled_data

def permuted_ND_worker(params):
    """
    Worker function for parallel execution of permuted_ND.
    Parameters:
        params (tuple): A tuple containing N, observable_results, and distance.
    Returns:
        numpy array: An array with permuted ND values for each permutation.
            - The shape of the array is (N, 5), where 5 corresponds to the number of observables.
    """
    N, all_results, distance = params
    ND_all = np.zeros((N, 5))
    for n in range(N):
        shuffled_results = random_permutation(data = all_results)
        ND_all[n, :] = ND_value(shuffled_results, distance)
    return ND_all



