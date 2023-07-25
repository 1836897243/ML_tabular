import torch
import random
import numpy as np
import os
from WorkFlow import WorkFlow
import pandas as pd

# 设定随机种子
seed = 0
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

# 定义超参
batch_size = 128
hidden_dim = 128
shuffle = False
file_dir_list = ['dataset/adult/', 'dataset/helena/']
for fileDir in file_dir_list:
    workflow = WorkFlow(fileDir, batch_size, hidden_dim, shuffle)
    feature_num = workflow.get_feature_num()
    result = {
        'feature': [],
        'metrics': [],
    }
    # choose one featur
    for idx1 in range(feature_num):
        feature_list = [idx1]
        encoder = workflow.pre_train(feature_list, device=device)
        encoder, head = workflow.train(encoder, device=device)
        metric = workflow.eval(encoder, head, device=device)
        print(idx1)
        result['feature'].append(str(idx1))
        result['metrics'].append(metric)
    '''
    # choose two features
    for idx1 in range(feature_num):
        for idx2 in range(idx1+1, feature_num):
            feature_list = [idx1, idx2]
            encoder = workflow.pre_train(feature_list, device=device)
            encoder, head = workflow.train(encoder, device=device)
            metric = workflow.eval(encoder, head, device=device)
            print(f'({idx1},{idx2})')
            result['feature'].append(f'({idx1},{idx2})')
            result['metrics'].append(metric)
            '''
    #save to excel file
    df = pd.DataFrame(result)
    df.to_excel(fileDir+'data.xlsx', index=False)

