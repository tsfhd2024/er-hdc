from typing import Dict, Tuple, Union

import numpy as np

from tools.utils import look


def correct_checksum(
    classHV: np.ndarray, checksum: np.ndarray, threshold: Union[int, float] = 2
) -> np.ndarray:
    """
    Correct checksum errors in a classHV matrix based on a specified threshold.

    Parameters:
    - classHV (np.ndarray): Input classHV matrix.
    - checksum (np.ndarray): Checksum array for error detection.
    - threshold (Union[int, float], optional): Threshold for error correction. Default is 2.

    Returns:
    - np.ndarray: Corrected classHV matrix after applying error correction based on the threshold.
    """
    sums = classHV.sum(axis=0)
    index = np.where(np.abs(checksum - sums) > threshold)[0]
    if len(index) > 0:
        classHV[:, index] = 0
    return classHV


def correct_clusters(
    classHV: np.ndarray,
    D: Dict[str, Dict[Tuple, np.ndarray]],
    Th: int = 300,
    alpha: int = 3,
) -> np.ndarray:
    """
    Correct clusters in a classHV matrix based on specified thresholds and dictionary values.

    Parameters:
    - classHV (np.ndarray): Input classHV matrix.
    - D (Dict[Tuple, Dict[str, np.ndarray]]): Dictionary containing information about clusters.
    - Th (int): Threshold for error correction. Default is 300.
    - alpha (int): Scaling factor for standard deviation in error correction. Default is 3.

    Returns:
    - np.ndarray: Corrected classHV matrix after applying cluster error correction.
    """
    for i in range(len(classHV.T)):
        dicts = D[look(i, D.keys())]
        mean = dicts["m"]
        std = dicts["s"]
        index1 = np.where(classHV[:, i] - mean + alpha * std > Th)[0]
        index0 = np.where(classHV[:, i] - mean - alpha * std < -Th)[0]
        if len(index0) + len(index1) > 0:
            classHV[:, i] = 0
    return classHV
