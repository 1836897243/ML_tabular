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


def compute2save_similarity_regression(features, analyse: Analyse, choose_num, save_dir):
    result = {
        'features': [],
        'Dot': [],
        'Degree': [],
        'Manhattan': [],
        'Euclidean': []
    }
    if choose_num <= 0 or choose_num >= len(features):
        return
    feature_tuple_list = list(combinations(features, choose_num))
    for feature_tuple in feature_tuple_list:
        feature_str = '(' + ','.join(str(item) for item in feature_tuple) + ')'
        feature_list = list(feature_tuple)
        _Dot, _Degree, _Manhattan, _Euclidean = analyse.compute_similarity_regression(feature_list)
        result['features'].append(feature_str)
        result['Dot'].append(_Dot)
        result['Degree'].append(_Degree)
        result['Manhattan'].append(_Manhattan)
        result['Euclidean'].append(_Euclidean)
    df = pd.DataFrame(result)
    df.to_excel(save_dir + '/similarity.xlsx', index=False)


def predict2save_similarity_regression(features, analyse: Analyse, choose_num, file_dir):
    result = {
        'features': [],
        'Dot': [],
        'Degree': [],
        'Manhattan': [],
        'Euclidean': []
    }
    if choose_num <= 0 or choose_num >= len(features):
        return
    feature_tuple_list = list(combinations(features, choose_num))
    for feature_tuple in feature_tuple_list:

        feature_str = '(' + ','.join(str(item) for item in feature_tuple) + ')'
        feature_list = list(feature_tuple)
        _Dot, _Degree, _Manhattan, _Euclidean = analyse.compute_similarity_regression_predict(
            feature_list, file_dir+'/'+feature_str, device)
        result['features'].append(feature_str)
        result['Dot'].append(_Dot)
        result['Degree'].append(_Degree)
        result['Manhattan'].append(_Manhattan)
        result['Euclidean'].append(_Euclidean)

    df = pd.DataFrame(result)
    df.to_excel(file_dir + '/predicted_feature_similarity.xlsx', index=False)

FileIndex = 0
FileDir = file_features['fileDir'][0]
print(FileDir)
DataDir = FileDir + '/data'
try_mkdir(DataDir)
for index, (name, features_list) in enumerate(file_features['features'].items()):
    features = features_list[FileIndex]
    choosen_num = file_features['choosen_num'][index]
    feature_save_dir = DataDir + '/' + name
    try_mkdir(feature_save_dir)
    analysis = Analyse(FileDir + '/', batch_size, shuffle)
    compute2save_similarity_regression(features, analysis, choosen_num, feature_save_dir)
    predict2save_similarity_regression(features, analysis, choosen_num, feature_save_dir)
