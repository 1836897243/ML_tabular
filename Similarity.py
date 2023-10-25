import torch
import numpy as np


def Degree(array1: np.ndarray, array2: np.ndarray) -> float:
    norm1 = np.linalg.norm(array1)
    norm2 = np.linalg.norm(array2)
    return np.arccos(np.dot(array1, array2) / (norm1 * norm2)) / 3.1415926 * 180

def pearson_coefficient(arr1:np.array, arr2:np.array):
    correlation_matrix  = np.corrcoef(arr1, arr2)
    return correlation_matrix[0,1]

def getAdjacencyMatrix(S: np.array, categorical: bool):
    if categorical is True:
        S = S.astype(int)
        A = S[:, np.newaxis] - S
        A[A != 0] = 1
    else:
        A = (S[:, np.newaxis] - S) ** 2
    return A


def getDoubleCenteredMatrix(S: np.array, categorical: bool):
    A = getAdjacencyMatrix(S, categorical)
    N = S.shape[0]
    Asl = np.sum(A, axis=0)
    Ask = np.sum(A, axis=1)
    Askl = np.sum(A)
    A = A - Asl[:, np.newaxis] / N - Ask / N + Askl / (N * N)
    return A


def getDistanceCovariance(S: np.array, S_categorical: bool, Z: np.array, Z_categorical: bool):

    assert S.shape == Z.shape
    N = S.shape[0]
    ES = getDoubleCenteredMatrix(S, S_categorical)
    EZ = getDoubleCenteredMatrix(Z, Z_categorical)
    return np.sum(ES * EZ) / (N * N)


def getDistanceCorrelation(S: np.array, S_categorical: bool, Z: np.array, Z_categorical: bool):
    S = np.squeeze(S)
    Z = np.squeeze(Z)
    ZS = getDistanceCovariance(Z, Z_categorical, S, S_categorical)
    ZZ = getDistanceCovariance(Z, Z_categorical, Z, Z_categorical)
    SS = getDistanceCovariance(S, S_categorical, S, S_categorical)
    return 0 if ZZ * SS == 0 else ZS / np.sqrt(ZZ * SS)
