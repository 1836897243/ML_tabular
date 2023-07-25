import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from enum import Enum
from Models import Head
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F


class TaskType(Enum):
    regression = 1
    binclass = 2
    multiclass = 3


def CrossEntropyLossWithSigmoid():
    def custom_loss(output, target):
        return F.cross_entropy(output, target)

    return custom_loss


class CustomDataset(Dataset):
    def __init__(self, features, targets):
        # 转换为tensor
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        target = self.targets[index]
        return feature, target


def LoadInput(num_dir, n_num, cat_dir, n_cat):
    num_features = None
    cat_features = None
    if n_num > 0:
        num_features = np.load(num_dir)
    if n_cat > 0:
        cat_features = np.load(cat_dir)
    return num_features, cat_features


def LoadTarget(target_dir):
    targets = np.load(target_dir)
    return targets


def processFeature(num_train, cat_train, num_val, cat_val, num_test, cat_test):
    mean = np.mean(num_train, axis=0)
    std = np.std(num_train, axis=0)
    num_train = (num_train - mean) / std
    num_val = (num_val - mean) / std
    num_test = (num_test - mean) / std

    if cat_train is not None:
        for idx in range(cat_train.shape[1]):
            label_encoder = LabelEncoder()
            cat_train[:, idx] = label_encoder.fit_transform(cat_train[:, idx])
            cat_val[:, idx] = label_encoder.transform(cat_val[:, idx])
            cat_test[:, idx] = label_encoder.transform(cat_test[:, idx])

        train_features = np.hstack((num_train, cat_train))
        val_features = np.hstack((num_val, cat_val))
        test_features = np.hstack((num_test, cat_test))
    else:
        train_features = num_train
        val_features = num_val
        test_features = num_test
    train_features = train_features.astype(np.float32)
    val_features = val_features.astype(np.float32)
    test_features = test_features.astype(np.float32)
    return torch.tensor(train_features, dtype=torch.float32), torch.tensor(val_features, dtype=torch.float32),\
           torch.tensor(test_features, dtype=torch.float32), std


def processTarget(tar_train, tar_val, tar_test, task_type, n_class):
    if task_type == TaskType.regression:
        mean = np.mean(tar_train)
        std = np.std(tar_train)

        tar_train = (tar_train - mean) / std
        tar_val = (tar_val - mean) / std
        tar_test = (tar_test - mean) / std
        return torch.tensor(tar_train, dtype=torch.float32), torch.tensor(tar_val, dtype=torch.float32),\
               torch.tensor(tar_test, dtype=torch.float32), std
    if task_type == TaskType.multiclass:
        label_encoder = LabelEncoder()
        tar_train = label_encoder.fit_transform(tar_train)
        tar_train = F.one_hot(torch.tensor(tar_train), num_classes=n_class).float()

        tar_val = label_encoder.transform(tar_val)
        tar_val = F.one_hot(torch.tensor(tar_val), num_classes=n_class).float()

        tar_test = label_encoder.transform(tar_test)
        tar_test = F.one_hot(torch.tensor(tar_test), num_classes=n_class).float()
        return tar_train, tar_val, tar_test, -1
    else:
        return torch.tensor(tar_train, dtype=torch.float32), torch.tensor(tar_val, dtype=torch.float32), \
               torch.tensor(tar_test, dtype=torch.float32), -1


class LoaderContainer:
    def __init__(self, data_dir, shuffle):
        self.shuffle = shuffle
        info_json = data_dir + 'info.json'
        with open(info_json, 'r') as f:
            info = json.load(f)
            n_num_features = info['n_num_features']
            n_cat_features = info['n_cat_features']
            self.input_num = n_num_features + n_cat_features
            self.num_list = list(range(n_num_features))
            self.cat_list = list(range(n_num_features, n_num_features + n_cat_features))
            self.out_dim = 0
            if info['task_type'] == 'regression':
                self.task_type = TaskType.regression
                self.out_dim = 1
            elif info['task_type'] == 'binclass':
                self.task_type = TaskType.binclass
                self.out_dim = 1
            elif info['task_type'] == 'multiclass':
                self.task_type = TaskType.multiclass
                self.out_dim = info['n_classes']

            # feature
            num_train = data_dir + 'N_train.npy'
            cat_train = data_dir + 'C_train.npy'
            num_train, cat_train = LoadInput(num_train, n_num_features, cat_train, n_cat_features)

            num_val = data_dir + 'N_val.npy'
            cat_val = data_dir + 'C_val.npy'
            num_val, cat_val = LoadInput(num_val, n_num_features, cat_val, n_cat_features)

            num_test = data_dir + 'N_test.npy'
            cat_test = data_dir + 'C_test.npy'
            num_test, cat_test = LoadInput(num_test, n_num_features, cat_test, n_cat_features)
            self.train_features, self.val_features, self.test_features, self.feature_std = \
                processFeature(num_train, cat_train, num_val, cat_val, num_test, cat_test)

            # targets
            train_target_dir = data_dir + 'y_train.npy'
            self.train_targets = LoadTarget(train_target_dir)
            val_target_dir = data_dir + 'y_val.npy'
            self.val_targets = LoadTarget(val_target_dir)
            test_target_dir = data_dir + 'y_test.npy'
            self.test_targets = LoadTarget(test_target_dir)
            self.train_targets, self.val_targets, self.test_targets, self.target_std = \
                processTarget(self.train_targets, self.val_targets, self.test_targets, self.task_type, self.out_dim)

    def getTrainLoader(self, batch_size):
        train_dataset = CustomDataset(self.train_features, self.train_targets)
        return DataLoader(train_dataset, batch_size, self.shuffle)

    def getValLoader(self, batch_size):
        val_dataset = CustomDataset(self.val_features, self.val_targets)
        return DataLoader(val_dataset, batch_size, self.shuffle)

    def getTestLoader(self, batch_size):
        test_dataset = CustomDataset(self.test_features, self.test_targets)
        return DataLoader(test_dataset, batch_size, self.shuffle)

    def getPreTrainLoader(self, batch_size, index):
        train_features = np.copy(self.train_features)
        train_features[:, index] = 0
        train_targets = np.copy(self.train_features[:, index])


        # val
        val_features = np.copy(self.val_features)
        val_features[:, index] = 0

        val_targets = np.copy(self.val_features[:, index])
        if index in self.num_list:
            feature_std = self.feature_std[index]
        elif index in self.cat_list:
            feature_std = -1
            n_class = len(np.unique(train_targets.astype(np.int32)))#1
            # set as a class which not in origin colume
            train_features[:, index] = n_class
            val_features[:, index] = n_class
            train_targets = F.one_hot(torch.tensor(train_targets, dtype=int), num_classes=n_class).float()
            val_targets = F.one_hot(torch.tensor(val_targets, dtype=int), num_classes=n_class).float()
        pre_train_dataset = CustomDataset(train_features, train_targets)
        val_dataset = CustomDataset(val_features, val_targets)
        return DataLoader(pre_train_dataset, batch_size, self.shuffle), \
               DataLoader(val_dataset, batch_size, self.shuffle), feature_std

    def getTrainHeadAndLossFunc(self, hidden_dim):
        if self.task_type == TaskType.regression:
            return Head(hidden_dim, self.out_dim), nn.MSELoss()
        elif self.task_type == TaskType.binclass:
            return Head(hidden_dim, self.out_dim), nn.BCEWithLogitsLoss()
        elif self.task_type == TaskType.multiclass:
            return Head(hidden_dim, self.out_dim), CrossEntropyLossWithSigmoid()

    def getPreTrainHeadAndLossFunc(self, hidden_dim, feature_index):
        if feature_index in self.num_list:
            return Head(hidden_dim, 1), nn.MSELoss()
        elif feature_index in self.cat_list:
            train_features = np.copy(self.train_features)
            num_class = len(np.unique(train_features[:, feature_index]))
            return Head(hidden_dim, num_class), CrossEntropyLossWithSigmoid()

    def getTargetStd(self):
        return self.target_std

    def getInfo(self):
        return self.input_num, self.num_list, self.cat_list, self.task_type
