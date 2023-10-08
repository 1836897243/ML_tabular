import numpy as np

from LoaderContainer import LoaderContainer, TaskType
from torch.utils.data import Dataset, DataLoader
from Similarity import *
from TrainEvalFunc import eval
import os

'''
# convert numerical feature to categorical feature
def numerical2categorical(numerical_arr: np.array, n_cat: int):
    max_value = np.max(numerical_arr)
    min_value = np.min(numerical_arr)

    data_range = max_value - min_value
    class_range = data_range / n_cat
    normalized_arr = (numerical_arr - min_value) / class_range
    cat_arr = np.clip(normalized_arr.astype(int), 0, n_cat - 1)
    return cat_arr


# convert categorical feature to numerical feature
def numerical2categorical_with_max_min(numerical_arr: np.array, n_cat: int, max_value: float, min_value: float):
    data_range = max_value - min_value
    class_range = data_range / n_cat
    normalized_arr = (numerical_arr - min_value) / class_range
    cat_arr = np.clip(normalized_arr.astype(int), 0, n_cat - 1)
    return cat_arr
'''


class Analyse:
    def __init__(self, loader_container):
        self.loader_container = loader_container
        data_loader = self.loader_container.getTrainLoader()
        self.features = None
        self.targets = None
        for (inputs, targets) in data_loader:
            if self.features is None and self.targets is None:
                self.features = inputs.numpy()
                self.targets = targets.numpy()
            else:
                self.features = np.append(self.features, inputs.numpy(), axis=0)
                self.targets = np.append(self.targets, targets.numpy(), axis=0)
        _, self.num_list, self.cat_list, self.task_type = self.loader_container.getInfo()

        self.degree_mean_feature = {}
        self.mean_degree_feature = {}
        self.dis_correlation = {}

    def compute_degree_of_mean_numerical_features(self, feature_list):
        for index in feature_list:
            assert index in self.num_list

        # no feature(feature_list = [])
        if len(feature_list) == 0:
            return 0

        # only one feature
        if len(feature_list) == 1:
            mean_feature = self.features[:, feature_list]
        else:
            if len(self.features[:, feature_list]) == 0:
                print(feature_list)
            mean_feature = np.mean(self.features[:, feature_list], axis=1)

        mean_feature = mean_feature.flatten()
        return Degree(mean_feature, self.targets.flatten())


    def compute_distance_correlation(self, feature_list):
        if len(feature_list) == 0:
            return 0

        _, num_list, cat_list, task_type = self.loader_container.getInfo()
        target_categorical = True
        if task_type == TaskType.regression:
            target_categorical = False

        for index in feature_list:
            if index not in self.dis_correlation:
                self.dis_correlation[index] = getDistanceCorrelation(self.features[:, index], index in cat_list,
                                                                     self.targets, target_categorical)

        correlations = [self.dis_correlation[index] for index in feature_list]
        return np.mean(correlations)
