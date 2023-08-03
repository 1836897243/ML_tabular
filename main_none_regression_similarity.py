import torch
import numpy as np
import os
import pandas as pd
from WorkFlow import WorkFlow
from itertools import combinations
from Analyse import Analyse


# cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# Hyper parameters
batch_size = 128
hidden_dim = 128
shuffle = False

# feature info
file_features = {
    'fileDir': ['dataset/california_housing', 'dataset/adult', 'dataset/helena'],
    'features': {
        'features_for_1': [
            range(8),
            range(14),
            range(27),
        ],
        'features_better_for_2': [
            [1, 0, 2, 3],
            [10, 8, 9],
            [14, 23, 12, 25],
        ],
        'features_worse_for_2': [
            [4, 6, 5],
            [1, 4, 5, 12, 2],
            [10, 5, 7, 22, 8],
        ],

        'features_better_for_3': [
            [0, 1, 2, 3, 7],
            [2, 6, 8, 9, 10],
            [10, 12, 14, 22, 23],
        ],
        'features_worse_for_3': [
            [0, 3, 4, 5, 6],
            [1, 2, 4, 5, 12],
            [5, 7, 8, 10, 25],
        ]
    },
    'choosen_num': [1, 2, 2, 3, 3]
}


def try_mkdir(dir2make):
    try:
        os.mkdir(dir2make)
    except FileExistsError:
        print(f"相对路径目录'{dir2make}'已经存在。")
    except Exception as e:
        print(f"创建相对路径目录'{dir2make}'时发生错误：{e}")


def compute2save_similarity_none_regression(feature_num, analyse: Analyse, save_dir):
    for feature_index in range(feature_num):
        result = {
            'class_value': [],
            'class_count': [],
            'mode_feature_value': [],
            'mode_feature_count': []
        }
        class_index, class_count, mode_feature_value, mode_feature_count = \
            analyse.compute_similarity_none_regression(feature_index)
        result['class_value'] = class_index
        result['class_count'] = class_count
        result['mode_feature_value'] = mode_feature_value
        result['mode_feature_count'] = mode_feature_count
        df = pd.DataFrame(result)
        df.to_excel(save_dir + '/'+str(feature_index)+'categorical_similarity.xlsx', index=False)


def predict2save_similarity_none_regression(feature_num, analyse: Analyse, save_dir):
    for feature_index in range(feature_num):
        result = {
            'class_value': [],
            'class_count': [],
            'mode_feature_value': [],
            'mode_feature_count': []
        }
        class_index, class_count, mode_feature_value, mode_feature_count = \
            analyse.compute_similarity_none_regression_predict(
                feature_index, save_dir + f'/features_for_1/({str(feature_index)})', device)
        result['class_value'] = class_index
        result['class_count'] = class_count
        result['mode_feature_value'] = mode_feature_value
        result['mode_feature_count'] = mode_feature_count
        df = pd.DataFrame(result)
        df.to_excel(save_dir + '/'+str(feature_index)+'predicted_categorical_similarity.xlsx', index=False)


# adult
FileDir = file_features['fileDir'][1]
print(FileDir)
DataDir = FileDir + '/data'
try_mkdir(DataDir)
# feature num of adult
feature_num = 14
try_mkdir(DataDir)
analysis = Analyse(FileDir + '/', batch_size, shuffle)
compute2save_similarity_none_regression(feature_num, analysis, DataDir)
predict2save_similarity_none_regression(feature_num, analysis, DataDir)

# helena
FileDir = file_features['fileDir'][2]
print(FileDir)
DataDir = FileDir + '/data'
try_mkdir(DataDir)
# feature num of helena
feature_num = 27
try_mkdir(DataDir)
analysis = Analyse(FileDir + '/', batch_size, shuffle)
compute2save_similarity_none_regression(feature_num, analysis, DataDir)
predict2save_similarity_none_regression(feature_num, analysis, DataDir)


