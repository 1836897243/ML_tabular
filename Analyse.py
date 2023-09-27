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

    def eval_from_dir(self, directory: str):
        # cuda
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        test = []
        val = []
        # encoder.pt and head.pt were saved by train_save_eval_models.py
        contents = os.listdir(directory)
        for item in contents:
            if os.path.isdir(directory + item):
                files = os.listdir(directory + item)
                if 'encoder.pt' not in files or 'head.pt' not in files:
                    break
                print(directory + item)
                trained_encoder = torch.load(directory + item + '/encoder.pt')
                trained_head = torch.load(directory + item + '/head.pt')
                test_data, val_data = eval(self.loader_container, trained_encoder, trained_head, device)
                test.append(test_data)
                val.append(val_data)
        return test, val

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
        return Degree(mean_feature, self.targets)

    def compute_mean_degree_of_numerical_features(self, feature_list):
        if len(feature_list) == 0:
            return 0

        for index in feature_list:
            assert index in self.num_list
            if index not in self.mean_degree_feature:
                self.mean_degree_feature[index] = Degree(self.features[:, index], self.targets)

        # only one feature
        degrees = [self.mean_degree_feature[index] for index in feature_list]
        return np.mean(degrees)

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

    '''
    def compute_similarity_regression_predict(self, feature_list, file_dir, device):
        for index in feature_list:
            assert index in self.num_list
        encoder = torch.load(file_dir+'/encoder.pt')
        predicted_features = None
        for feature_index in feature_list:
            head = torch.load(file_dir+'/' + str(feature_index) + 'feature_head.pt')
            data_loader, _1, _2 = self.loader_container.getPreTrainLoader(feature_index)

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
    
    
    def compute_similarity_none_regression_predict(self, feature_index, file_dir, device):
        assert self.task_type == TaskType.binclass or self.task_type == TaskType.multiclass

        encoder = torch.load(file_dir+'/encoder.pt')

        head = torch.load(file_dir+'/' + str(feature_index) + 'feature_head.pt')
        data_loader, _1, _2 = self.loader_container.getPreTrainLoader(feature_index)
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
    '''
