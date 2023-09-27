import random

import numpy as np
import pandas as pd
from Analyse import Analyse
from LoaderContainer import LoaderContainer
import torch.multiprocessing
import time
from train_save_eval_models import eval_from_dir, train_and_save_one_model
from train_save_eval_models import setRandomSeed
def generate_feature_combination_regression(loader_container: LoaderContainer, count, lower_val, upper_val):
    analysis = Analyse(loader_container)
    # generate feature_list
    feature_list_list = []
    _degress = []
    _1, num_list, cat_list, _2 = loader_container.getInfo()
    while len(feature_list_list) < count:
        feature_num = random.randint(1, 10)#len(num_list))
        feature_list = random.sample(range(0, len(num_list)), feature_num)
        feature_list.sort()
        _Degree = analysis.compute_degree_of_mean_numerical_features(feature_list)
        if upper_val > _Degree > lower_val and feature_list not in feature_list_list:
            feature_list_list.append(feature_list)
            _degress.append(_Degree)
    feature_list_list.append([])
    _degress.sort()
    print(feature_list_list)
    return feature_list_list


def train_and_save(dataset_dir, encoder_type, feature_list_list,dir_name2save,
                   seed, batch_size, hidden_dim, shuffle):

    for feature_list in feature_list_list:
        print(f'current feature_list:{feature_list}')
        train_and_save_one_model(dataset_dir, dir_name2save, encoder_type, feature_list,
                                                seed, batch_size, hidden_dim, shuffle)
        '''
    processes = []
    for feature_list in feature_list_list:
        process = torch.multiprocessing.Process(target=train_and_save_one_model,
                                          args=(dataset_dir, dir_name2save, encoder_type, feature_list,
                                                seed, batch_size, hidden_dim, shuffle))
        processes.append(process)
        process.start()

        # waitting for process finish
        for process in processes:
            process.join()
        '''


Seed = 0
# Hyper parameters
Batch_size = 128
Hidden_dim = 128
Shuffle = False

if __name__ == '__main__':
    setRandomSeed(0)
    '''only for regression task and numerical features, if there are categorical features, check code in model'''
    dataset_dir = 'dataset/regression/superconductivty_data/'
    loader_container = LoaderContainer(dataset_dir=dataset_dir, batch_size=Batch_size, shuffle=Shuffle)
    count = 100
    lower_val = 0
    upper_val = 360
    encoder_types = ['MLP', "ResNet"]

    start = time.perf_counter()
    # get feature combinations
    feature_list_list = generate_feature_combination_regression(
        loader_container, count=count, lower_val=lower_val, upper_val=upper_val)
    for encoder_type in encoder_types:
        dir_name2save = 'QT(' + encoder_type + ')' + str(lower_val) + '-' + str(upper_val) + '(Degree)w_reg'

        # train models and save
        train_and_save(dataset_dir=dataset_dir, encoder_type=encoder_type, feature_list_list=feature_list_list,
                       dir_name2save=dir_name2save, seed=Seed, batch_size=Batch_size, hidden_dim=Hidden_dim, shuffle=Shuffle)

        # eval the model and compute similarity, finally save to excel
        computed_data_dir = dataset_dir + dir_name2save + '/'
        test, val, epoch_num_pre_train, epoch_num_train, feature_list_list = eval_from_dir(computed_data_dir, loader_container)

        analyse = Analyse(loader_container)
        degree = [analyse.compute_degree_of_mean_numerical_features(feature_list) for feature_list in feature_list_list]

        DistanceCorrelations = [analyse.compute_distance_correlation(feature_list)
                                for feature_list in feature_list_list]
        # save data to excel
        result = {
            'feature': feature_list_list,
            'test_metrics': test,
            'val_metrics': val,
            'Degree': degree,
            'epcoh_num_pre_train': epoch_num_pre_train,
            'epoch_num_train': epoch_num_train,
            'DistanceCorrelation': DistanceCorrelations
        }
        df = pd.DataFrame(result)
        df.to_excel(computed_data_dir+'data.xlsx', index=False)

    # print cost of time
    end = time.perf_counter()
    print("运行耗时", end - start)




