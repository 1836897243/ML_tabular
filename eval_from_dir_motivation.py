import torch
import numpy as np
import torch.optim as optim
import os
from matplotlib import pyplot as plt
import pandas as pd
from TrainEvalFunc import eval, RMSE
import argparse
from LoaderContainer import LoaderContainer
from Analyse import Analyse

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
def is_feature_dir(s):
    if len(s)<2 or s[0]!='[' or s[-1]!=']':
        return False, 0
    if s == '[]':
        return True, 0
    is_num = True
    nums = s[1:-1].split(',')
    for num in nums:
        is_num = is_num and is_int(num)
    return is_num, len(nums)

def eval_from_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='<Required>file directory of dataset', required=True)
    parser.add_argument('--models_dir', type=str,
                        help='<Required> directory of saved models', required=True)
    parser.add_argument('--save_dir', type=str,
                        help='<Required> name of directory to save computed data', required=True)
    parser.add_argument('--feature_num', type=int, help='batch_size when train', default=0)
    parser.add_argument('--save_file_name', type=str, help='<Required>file to save data', required=True)


    args = parser.parse_args()
    device = torch.device("cpu")
    dataset_dir = args.dataset_dir
    save_file_name = args.save_file_name
    directory = args.models_dir
    feature_num = args.feature_num
    save_dir = args.save_dir

    loader_container = LoaderContainer(dataset_dir=dataset_dir, batch_size=128, shuffle=False, scaler_type='StandardScaler')
    test = []
    val = []
    feature_train = []
    feature_test = []
    feature_val = []
    epochs_num_train = []
    feature_list_list = []
    # encoder.pt and head.pt were saved by train_save_eval_models.py
    contents = os.listdir(directory)
    for item in contents:
        cur_dir = os.path.join(directory, item)
        if os.path.isdir(cur_dir):
            
            files = os.listdir(cur_dir)
            if 'encoder.pt' not in files or 'head.pt' not in files:
                continue
               

            # pretraing info
            is_feature, feature_nums = is_feature_dir(item)
            if is_feature and (feature_nums == feature_num or feature_nums == 0):
            
                if len(item) > 2:
                    feature_list = [int(feature) for feature in item[1:-1].split(',')]
                else:
                    feature_list = []
                feature_list_list.append(feature_list)
                trained_encoder = torch.load(os.path.join(cur_dir, 'encoder.pt'), map_location='cpu')
                trained_head = torch.load(os.path.join(cur_dir, 'head.pt'), map_location='cpu')
                test_data, val_data = eval(loader_container, trained_encoder, trained_head, device)
                
                feature_encoder = trained_encoder#torch.load(os.path.join(cur_dir, 'feature_encoder.pt'), map_location='cpu')
                feature_train_loss = 0
                feature_val_loss = 0
                feature_test_loss = 0
                for file in files:
                    if file[-15:] != 'feature_head.pt':
                        continue
                    feature_index = int(file[:-16])
                    feature_head = torch.load(os.path.join(cur_dir, file), map_location='cpu')
                    train_pretrain_loader, val_pretrain_loader, test_pretrain_loader \
                        = loader_container.getPreTrainLoader(feature_index)

                    feature_train_loss += RMSE(data_loader=train_pretrain_loader, encoder=feature_encoder,
                                               head=feature_head, inverse_transform_func=None, device=device)
                    feature_val_loss += RMSE(data_loader=test_pretrain_loader, encoder=feature_encoder,
                                               head=feature_head, inverse_transform_func=None, device=device)
                    feature_test_loss += RMSE(data_loader=val_pretrain_loader, encoder=feature_encoder,
                                              head=feature_head, inverse_transform_func=None, device=device)

                feature_train.append(feature_train_loss)
                feature_val.append(feature_val_loss)
                feature_test.append(feature_test_loss)

                # epoch_info
                epoch_info_file_name = 'epoch_info.csv'
                epochs_info = np.loadtxt(os.path.join(cur_dir, epoch_info_file_name), delimiter=',')
                test.append(test_data)
                val.append(val_data)

    

    # for file which dose not contain feature_encoder and feature_head
    if len(feature_test) == 0:
        feature_test = np.zeros(len(test))
        feature_val = np.zeros(len(test))
        feature_train = np.zeros(len(test))
    
    
    
    analyse = Analyse(loader_container)
    degree = [analyse.compute_degree_of_mean_numerical_features(feature_list) for feature_list in feature_list_list]

    DistanceCorrelations = [analyse.compute_distance_correlation(feature_list)
                           for feature_list in feature_list_list]
    PearsonCoefficients = [analyse.compute_pearson_coefficient(feature_list)
                            for feature_list in feature_list_list]

    


    # save data to excel
    result = {
        'feature': feature_list_list,
        'test_motivations': test,
        'val_motivations': val,
        'feature_test': feature_test,
        'feature_val': feature_val,
        'feature_train': feature_train,
        'Degree': degree,
        'DistanceCorrelation': DistanceCorrelations,
        'PearsonCoefficient': PearsonCoefficients
    }
    df = pd.DataFrame(result)
    df.to_excel(os.path.join(save_dir, save_file_name + '.xlsx'), index=False)
    return

if __name__ == '__main__':
    eval_from_dir()



