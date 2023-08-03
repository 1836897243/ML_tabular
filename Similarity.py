import torch
import numpy as np


def Dot(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    return np.dot(array1, array2)


def Degree(array1: np.ndarray, array2: np.ndarray) -> float:
    norm1 = np.linalg.norm(array1)
    norm2 = np.linalg.norm(array2)
    return np.arccos(np.dot(array1, array2)/(norm1 * norm2))


def ManhattanDistance(array1: np.ndarray, array2: np.ndarray) -> float:
    return np.linalg.norm(array1 - array2, ord=1)


def EuclideanDistance(array1: np.ndarray, array2: np.ndarray) -> float:
    return np.linalg.norm(array1 - array2)


def CatSimilarity(array: np.ndarray[int], target: np.ndarray[int], n_class: int) -> \
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    classes = []
    class_counts = []
    mode_feature = []
    mode_value_counts = []
    for a_class in range(n_class):
        indexes = np.where(target == a_class)
        arr = array[indexes]

        # get unique value and counts
        unique_values, counts = np.unique(arr, return_counts=True)
        # find the index of num appear most
        mode_index = np.argmax(counts)
        # get the the num appear most
        mode_value = unique_values[mode_index]
        # get its count
        mode_count = counts[mode_index]

        classes.append(a_class)
        class_counts.append(len(arr))
        mode_feature.append(mode_value)
        mode_value_counts.append(mode_count)
    return classes, class_counts, mode_feature, mode_value_counts
