import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from enum import Enum
from Models import Head
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler, MinMaxScaler
import torch.nn.functional as F


class TaskType(Enum):
    regression = 1
    binclass = 2
    multiclass = 3


def CrossEntropyLossWithSoftmax():
    def custom_loss(output, target):
        return F.cross_entropy(output, target)

    return custom_loss


class NRMSELoss(nn.Module):
    def __init__(self):
        super(NRMSELoss, self).__init__()

    def forward(self, predicted, target):
        # 计算均方误差
        nrmse_loss = torch.sqrt(torch.mean((predicted - target) ** 2))
        return nrmse_loss


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


def processFeature(num_train, cat_train, num_val, cat_val, num_test, cat_test, scaler_type: str):
    assert scaler_type in ['QuantileTransformer', 'StandardScaler', 'MinMaxScaler']
    if scaler_type == 'QuantileTransformer':
        scalar = QuantileTransformer()
    elif scaler_type == 'StandardScaler':
        scalar = StandardScaler()
    else:
        scalar = MinMaxScaler()
    num_train = scalar.fit_transform(num_train)
    num_val = scalar.transform(num_val)
    num_test = scalar.transform(num_test)

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
    return torch.tensor(train_features, dtype=torch.float32), torch.tensor(val_features, dtype=torch.float32), \
        torch.tensor(test_features, dtype=torch.float32)


def processTarget(tar_train, tar_val, tar_test, task_type, n_class, scaler_type: str):
    if task_type == TaskType.regression:

        assert scaler_type in ['QuantileTransformer', 'StandardScaler', 'MinMaxScaler']
        if scaler_type == 'QuantileTransformer':
            scalar = QuantileTransformer()
        elif scaler_type == 'StandardScaler':
            scalar = StandardScaler()
        else:
            scalar = MinMaxScaler()
        tar_train = scalar.fit_transform(tar_train.reshape(-1, 1))
        tar_val = scalar.transform(tar_val.reshape(-1, 1))
        tar_test = scalar.transform(tar_test.reshape(-1, 1))

        return torch.tensor(tar_train, dtype=torch.float32), torch.tensor(tar_val, dtype=torch.float32), \
            torch.tensor(tar_test, dtype=torch.float32), scalar.inverse_transform

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
    def __init__(self, dataset_dir, batch_size, shuffle, scaler_type):
        self.shuffle = shuffle
        self.batch_size = batch_size
        info_json = dataset_dir + 'info.json'
        with open(info_json, 'r') as f:
            info = json.load(f)
            n_num_features = info['n_num_features']
            n_cat_features = info['n_cat_features']
            self.input_num = n_num_features + n_cat_features
            self.num_list = list(range(0, n_num_features))
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
            num_train = dataset_dir + 'N_train.npy'
            cat_train = dataset_dir + 'C_train.npy'
            num_train, cat_train = LoadInput(num_train, n_num_features, cat_train, n_cat_features)

            num_val = dataset_dir + 'N_val.npy'
            cat_val = dataset_dir + 'C_val.npy'
            num_val, cat_val = LoadInput(num_val, n_num_features, cat_val, n_cat_features)

            num_test = dataset_dir + 'N_test.npy'
            cat_test = dataset_dir + 'C_test.npy'
            num_test, cat_test = LoadInput(num_test, n_num_features, cat_test, n_cat_features)

            # targets
            train_target_dir = dataset_dir + 'y_train.npy'
            train_targets = LoadTarget(train_target_dir)
            val_target_dir = dataset_dir + 'y_val.npy'
            val_targets = LoadTarget(val_target_dir)
            test_target_dir = dataset_dir + 'y_test.npy'
            test_targets = LoadTarget(test_target_dir)

            '''
            # shuffle  val and test dataset
            print(f'num_train.shape is {num_train.shape}')
            print(f'num_val.shape is {num_val.shape}')
            print(f'num_test.shape is {num_test.shape}')
            all_num_feature = np.vstack((num_train, num_val, num_test))
            print(f'num_feature.shape is {all_num_feature.shape}')
            if cat_train is not None:
                print(f'cat_train.shape is {cat_train.shape}')
                print(f'cat_val.shape is {cat_val.shape}')
                print(f'cat_test.shape is {cat_test.shape}')
                all_cat_feature = np.vstack((cat_train, cat_val, cat_test))
                print(f'cat_feature.shape is {all_cat_feature.shape}')

            print(f'train_targets.shape is {train_targets.shape}')
            print(f'val_targets.shape is {val_targets.shape}')
            print(f'test_targets.shape is {test_targets.shape}')
            all_target = np.concatenate((train_targets, val_targets, test_targets), axis=0)
            print(f'all_target.shape is {all_target.shape}')

            all_length = len(all_target)
            train_length = len(num_train)
            val_length = len(num_val)
            test_length = len(num_test)

            split_index1 = train_length
            split_index2 = train_length + val_length

            indices = torch.randperm(all_length)
            shuffled_num_feature = all_num_feature[indices]
            num_train = shuffled_num_feature[:split_index1]
            num_val = shuffled_num_feature[split_index1:split_index2]
            num_test = shuffled_num_feature[split_index2:]

            if cat_train is not None:
                shuffled_cat_feature = all_cat_feature[indices]
                cat_train = shuffled_cat_feature[:split_index1]
                cat_val = shuffled_cat_feature[split_index1:split_index2]
                cat_test = shuffled_cat_feature[split_index2:]

            shuffled_target = all_target[indices]
            train_targets = shuffled_target[:split_index1]
            val_targets = shuffled_target[split_index1:split_index2]
            test_targets = shuffled_target[split_index2:]
            '''
            self.train_features, self.val_features, self.test_features, = \
                processFeature(num_train, cat_train, num_val, cat_val, num_test, cat_test, scaler_type)

            self.train_targets, self.val_targets, self.test_targets, self.inverse_transform_func = \
                processTarget(train_targets, val_targets, test_targets, self.task_type, self.out_dim, scaler_type)

    def getTrainLoader(self):
        train_dataset = CustomDataset(self.train_features, self.train_targets)
        return DataLoader(train_dataset, self.batch_size, self.shuffle)

    def getValLoader(self):
        val_dataset = CustomDataset(self.val_features, self.val_targets)
        return DataLoader(val_dataset, self.batch_size, self.shuffle)

    def getTestLoader(self):
        test_dataset = CustomDataset(self.test_features, self.test_targets)
        return DataLoader(test_dataset, self.batch_size, self.shuffle)

    def getPreTrainLoader(self, index):
        # [:self.batch_size*batch_num] is used to confine batch num when pretrain
        # batch_num = 5
        train_features = np.copy(self.train_features)  # [:self.batch_size*batch_num]
        train_targets = np.copy(self.train_features[:, index])  # [:self.batch_size*batch_num]
        # val
        val_features = np.copy(self.val_features)
        val_targets = np.copy(self.val_features[:, index])

        test_features = np.copy(self.test_features)
        test_targets = np.copy(self.test_features[:, index])
        if index in self.num_list:
            # if feature is numerical, set default value as 0
            train_features[:, index] = 0
            val_features[:, index] = 0
            test_features[:, index] = 0
        elif index in self.cat_list:
            n_class = len(np.unique(train_targets.astype(np.int32)))
            # if the feature is categorical, set default value as a type not in dataset
            train_features[:, index] = n_class
            val_features[:, index] = n_class
            test_features[:, index] = n_class
            train_targets = F.one_hot(torch.tensor(train_targets).long(), num_classes=n_class).float()
            val_targets = F.one_hot(torch.tensor(val_targets).long(), num_classes=n_class).float()
            test_targets = F.one_hot(torch.tensor(test_targets).long(), num_class=n_class).float()
        pre_train_dataset = CustomDataset(train_features, train_targets)
        val_dataset = CustomDataset(val_features, val_targets)
        test_dataset = CustomDataset(test_features, test_targets)
        return (DataLoader(pre_train_dataset, self.batch_size, self.shuffle),
                DataLoader(val_dataset, self.batch_size, self.shuffle),
                DataLoader(test_dataset, self.batch_size, self.shuffle))

    def getTrainHeadAndLossFunc(self, hidden_dim):
        if self.task_type == TaskType.regression:
            return Head(hidden_dim, self.out_dim), nn.MSELoss()
        elif self.task_type == TaskType.binclass:
            return Head(hidden_dim, self.out_dim), nn.BCEWithLogitsLoss()
        elif self.task_type == TaskType.multiclass:
            return Head(hidden_dim, self.out_dim), CrossEntropyLossWithSoftmax()

    def getPreTrainHeadAndLossFunc(self, hidden_dim, feature_index):
        if feature_index in self.num_list:
            return Head(hidden_dim, 1), NRMSELoss()
        elif feature_index in self.cat_list:
            train_features = np.copy(self.train_features)
            num_class = len(np.unique(train_features[:, feature_index]))
            return Head(hidden_dim, num_class), CrossEntropyLossWithSoftmax()

    def getAllPreTrainHeadAndLossFuncList(self, hidden_dim):
        all_pre_train_head_list = []
        all_pre_train_loss_func_list = []
        feature_num = len(self.cat_list) + len(self.num_list)
        for feature_index in range(feature_num):
            if feature_index in self.num_list:
                all_pre_train_head_list.append(Head(hidden_dim, 1))
                all_pre_train_loss_func_list.append(NRMSELoss())
            elif feature_index in self.cat_list:
                train_features = np.copy(self.train_features)
                num_class = len(np.unique(train_features[:, feature_index]))
                all_pre_train_head_list.append(Head(hidden_dim, num_class))
                all_pre_train_loss_func_list.append(CrossEntropyLossWithSoftmax())
        return all_pre_train_head_list, all_pre_train_loss_func_list

    def getInverseTransformFunc(self):
        return self.inverse_transform_func

    def getInfo(self):
        return self.input_num, self.num_list, self.cat_list, self.task_type

    def getOutDim(self):
        return self.out_dim
