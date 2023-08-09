import torch
import random
import numpy as np
import os
from WorkFlow import WorkFlow
import pandas as pd
from itertools import combinations
from Analyse import Analyse

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
shuffle = False


def try_mkdir(dir2make):
    try:
        os.mkdir(dir2make)
    except FileExistsError:
        print(f"相对路径目录'{dir2make}'已经存在。")
    except Exception as e:
        print(f"创建相对路径目录'{dir2make}'时发生错误：{e}")

def train_and_save(workflow, analysis, feature_list_list, seed, save_file_dir):

    result = {
        'feature': [],
        'test_metrics': [],
        'val_metrics': [],
        'Degree': []
    }

    for feature_list in feature_list_list:
        setRandomSeed(seed)
        encoder, feature_heads = workflow.pre_train(feature_list, device=device)
        encoder, head = workflow.train(encoder, device=device)
        test_metric, val_metric = workflow.eval(encoder, head, device=device)

        feature_str = '(' + ','.join(str(item) for item in feature_list)+')'
        result['feature'].append(feature_str)
        result['test_metrics'].append(test_metric)
        result['val_metrics'].append(val_metric)

        _Dot, _Degree, _Manhattan, _Euclidean = analysis.compute_similarity_regression(feature_list)
        result['Degree'].append(_Degree)
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
    df.to_excel(save_file_dir+'/data_degree_lower80.xlsx', index=False)


def main():
    filedir = 'dataset/california_housing'

    analysis = Analyse(filedir + '/', batch_size, shuffle)
    # generate feature_list
    feature_list_list = []
    count = 40
    while len(feature_list_list) < count:
        feature_num = random.randint(0, 7)
        feature_list = random.sample(range(0, 8), feature_num)
        _Dot, _Degree, _Manhattan, _Euclidean = analysis.compute_similarity_regression(feature_list)
        if _Degree < 80 and feature_list not in feature_list_list:
            feature_list_list.append(feature_list)
            print(len(feature_list_list))
    print(f'len(feature_list_list)={len(feature_list_list)}')
    workflow = WorkFlow(filedir+'/', batch_size, hidden_dim, shuffle)
    data_dir = filedir+'/data/lower80'
    try_mkdir(data_dir)
    print(data_dir)
    train_and_save(workflow, analysis, feature_list_list, Seed, data_dir)

main()







