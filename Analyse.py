import numpy as np

from LoaderContainer import LoaderContainer, TaskType
from torch.utils.data import Dataset, DataLoader
from Similarity import *
from TrainEvalFunc import get_predicted_multi_classification, get_predicted_regression


def numerical2categorical(numerical_arr: np.array, n_cat: int):
    max_value = np.max(numerical_arr)
    min_value = np.min(numerical_arr)

    data_range = max_value - min_value
    class_range = data_range / n_cat
    normalized_arr = (numerical_arr - min_value) / class_range
    cat_arr = np.clip(normalized_arr.astype(int), 0, n_cat - 1)
    return cat_arr


def numerical2categorical_with_max_min(numerical_arr: np.array, n_cat: int, max_value: float, min_value: float):
    data_range = max_value - min_value
    class_range = data_range / n_cat
    normalized_arr = (numerical_arr - min_value) / class_range
    cat_arr = np.clip(normalized_arr.astype(int), 0, n_cat - 1)
    return cat_arr


class Analyse:
    def __init__(self, file_dir: str, batch_size: int, shuffle: bool):
        self.batch_size = batch_size
        self.loader_container = LoaderContainer(file_dir, shuffle)
        data_loader = self.loader_container.getTrainLoader(self.batch_size)
        self.features = None
        self.targets = None
        for (inputs, targets) in data_loader:
            if self.features is None and self.targets is None:
                self.features = inputs.numpy()
                self.targets = targets.numpy()
            else:
                self.features = np.append(self.features, inputs.numpy(), axis=0)
                self.targets = np.append(self.targets, targets.numpy(), axis=0)
        print(f'shape of train.features is {self.features.shape}')
        print(f'shape of train.targets is {self.targets.shape}')
        _, self.num_list, self.cat_list, self.task_type = self.loader_container.getInfo()

    def compute_similarity_regression_predict(self, feature_list, file_dir, device):
        for index in feature_list:
            assert index in self.num_list
        encoder = torch.load(file_dir+'/encoder.pt')
        predicted_features = None
        for feature_index in feature_list:
            head = torch.load(file_dir+'/' + str(feature_index) + 'feature_head.pt')
            data_loader, _1, _2 = self.loader_container.getPreTrainLoader(self.batch_size, feature_index)

            predicted = get_predicted_regression(encoder, head, data_loader, device)

            if predicted_features is None:
                predicted_features = predicted
            else:
                predicted_features = np.vstack((predicted_features, predicted))

        if len(feature_list) != 1:
            predicted_features = np.transpose(predicted_features)
            mean_feature = np.mean(predicted_features, axis=1)
        else:
            mean_feature = predicted_features

        return Dot(mean_feature, self.targets), Degree(mean_feature, self.targets), \
               ManhattanDistance(mean_feature, self.targets), EuclideanDistance(mean_feature, self.targets)

    # for one feature or multi feature
    def compute_similarity_regression(self, feature_list):
        for index in feature_list:
            assert index in self.num_list

        # only one feature
        if len(feature_list) == 1:
            mean_feature = self.features[:, feature_list]
        else:
            mean_feature = np.mean(self.features[:, feature_list], axis=1)

        mean_feature = mean_feature.flatten()
        return Dot(mean_feature, self.targets), Degree(mean_feature, self.targets), \
               ManhattanDistance(mean_feature, self.targets), EuclideanDistance(mean_feature, self.targets)

    def compute_similarity_none_regression_predict(self, feature_index, file_dir, device):
        assert self.task_type == TaskType.binclass or self.task_type == TaskType.multiclass

        encoder = torch.load(file_dir+'/encoder.pt')

        head = torch.load(file_dir+'/' + str(feature_index) + 'feature_head.pt')
        data_loader, _1, _2 = self.loader_container.getPreTrainLoader(self.batch_size, feature_index)
        cat_feature = None

        if feature_index in self.num_list:
            predicted = get_predicted_regression(encoder, head, data_loader, device)
            max_value = np.max(self.features[:, feature_index])
            min_value = np.min(self.features[:, feature_index])
            if self.task_type == TaskType.binclass:
                cat_feature = numerical2categorical_with_max_min(
                    predicted, 2, max_value, min_value)
            elif self.task_type == TaskType.multiclass:
                cat_feature = numerical2categorical_with_max_min(
                    predicted, self.loader_container.getOutDim(), max_value, min_value)
        else:
            predicted = get_predicted_multi_classification(encoder, head, data_loader, device)
            cat_feature = predicted.astype(int)

        # check accuracy
        feature_in_train = None
        if feature_index in self.num_list:
            if self.task_type == TaskType.binclass:
                feature_in_train = numerical2categorical(self.features[:, feature_index], 2)
            elif self.task_type == TaskType.multiclass:
                feature_in_train = numerical2categorical(self.features[:, feature_index],
                                                         self.loader_container.getOutDim())
        else:
            feature_in_train = self.features[:, feature_index].astype(int)
        corrects = (cat_feature == feature_in_train).sum()
        total = len(cat_feature)
        print(f'accuracy for {feature_index} is {corrects/total} ')

        if self.task_type == TaskType.binclass:
            return CatSimilarity(cat_feature, self.targets, 2)
        elif self.task_type == TaskType.multiclass:
            scalar_target = np.argmax(self.targets, axis=1)
            return CatSimilarity(cat_feature, scalar_target, self.loader_container.getOutDim())

    # for one feature
    def compute_similarity_none_regression(self, feature_index):
        assert self.task_type == TaskType.binclass or self.task_type == TaskType.multiclass

        if feature_index in self.num_list:
            if self.task_type == TaskType.binclass:
                cat_feature = numerical2categorical(self.features[:, feature_index], 2)
            elif self.task_type == TaskType.multiclass:
                cat_feature = numerical2categorical(self.features[:, feature_index], self.loader_container.getOutDim())
        else:
            cat_feature = self.features[:, feature_index].astype(int)

        if self.task_type == TaskType.binclass:
            return CatSimilarity(cat_feature, self.targets, 2)
        elif self.task_type == TaskType.multiclass:
            scalar_target = np.argmax(self.targets, axis=1)
            return CatSimilarity(cat_feature, scalar_target, self.loader_container.getOutDim())


