import torch
import random
import numpy as np
import os
from WorkFlow import WorkFlow
import pandas as pd
from itertools import combinations
# 设定随机种子
Seed = 0


def setRandomSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


# cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Hyper parameters
batch_size = 128
hidden_dim = 128
shuffle = True


# feature info
file_features ={
    'fileDir':  ['dataset/california_housing', 'dataset/adult', 'dataset/helena'],
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


def train_and_save(workflow, features, choose_num, seed, save_file_dir):

    result = {
        'feature': [],
        'test_metrics': [],
        'val_metrics': []
    }
    if choose_num <= 0 or choose_num >= len(features):
        return
    feature_tuple_list = list(combinations(features, choose_num))
    if choose_num == 1:
        feature_tuple_list.append([])

    for feature_tuple in feature_tuple_list:
        feature_list = list(feature_tuple)
        setRandomSeed(seed)
        encoder, feature_heads = workflow.pre_train(feature_list, device=device)
        encoder, head = workflow.train(encoder, device=device)
        test_metric, val_metric = workflow.eval(encoder, head, device=device)

        feature_str = '(' + ','.join(str(item) for item in feature_tuple)+')'
        result['feature'].append(feature_str)
        result['test_metrics'].append(test_metric)
        result['val_metrics'].append(val_metric)

        # save model to excel file
        cur_feature_dir = save_file_dir+'/'+feature_str
        try_mkdir(cur_feature_dir)
        cur_feature_dir = cur_feature_dir+'/'
        torch.save(encoder, cur_feature_dir+'encoder' + '.pt')
        torch.save(head, cur_feature_dir+'head'+'.pt')
        for feature_head, feature_index in zip(feature_heads, feature_list):
            torch.save(feature_head, cur_feature_dir+str(feature_index)+'feature_head'+'.pt')

    # save data to excel file
    df = pd.DataFrame(result)
    df.to_excel(save_file_dir+'/data.xlsx', index=False)



for FileIndex, FileDir in enumerate(file_features['fileDir']):
    print(FileDir)
    DataDir = FileDir+'/data'
    try_mkdir(DataDir)
    setRandomSeed(Seed)
    workflow = WorkFlow(FileDir+'/', batch_size, hidden_dim, shuffle)
    for index, (name, features_list) in enumerate(file_features['features'].items()):
        features = features_list[FileIndex]
        choosen_num = file_features['choosen_num'][index]
        feature_save_dir = DataDir+'/'+name
        try_mkdir(feature_save_dir)
        train_and_save(workflow, features, choosen_num, Seed, feature_save_dir)






