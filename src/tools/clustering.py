import numpy as np
from sklearn.cluster import DBSCAN

from tools.utils import N_bits


def clustering(classHV, eps):
    """
    Performs clustering on the given classHV matrix using DBSCAN algorithm.

    Args:
        classHV (np.ndarray): The input matrix for clustering.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.

    Returns:
        Tuple[Dict, np.ndarray, Dict, int, None]: A tuple containing the following:
            - D (Dict): A dictionary representing the clustering results.
            - labels (np.ndarray): The labels assigned to each sample.
            - output_dict (Dict): A dictionary representing the clustering results.
            - num_clusters (int): The number of clusters found.
            - None: Placeholder value.

    Examples:
        >>> classHV = np.array([[1, 2, 3], [4, 5, 6]])
        >>> eps = 0.5
        >>> clustering(classHV, eps)
        ({(0,): {'m': array([1.5, 3.5]), 's': array([1.5, 1.5])},
        (1,): {'m': array([4.5, 5.5]), 's': array([1.5, 1.5])}},
        array([0, 0]),
        {0: {'m': array([1.5, 3.5]), 's': array([1.5, 1.5])}},
        1,
        None)
    """

    clustering = DBSCAN(eps=eps, min_samples=1, metric="cosine").fit(classHV.T)
    labels = np.array(clustering.labels_)
    unique = np.unique(labels, return_counts=True)
    D = {}
    output_dict = {}
    for l in unique[0]:
        index = np.where(labels == l)[0]
        mean = classHV[:, index].mean(axis=-1)
        std = classHV[:, index].std(axis=-1)
        mean = N_bits(mean)
        std = N_bits(std)
        D[tuple(index)] = {"m": mean, "s": std}
        output_dict[l] = {"m": mean, "s": std}
    return D, len(unique[0])


def process(classHV):
    return classHV.sum(axis=0)
