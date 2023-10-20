import torch
import numpy as np
import os
import argparse
from WorkFlow import WorkFlow

from UitlsTools import save_image, try_mkdir, setRandomSeed
from LoaderContainer import LoaderContainer

def Motivation_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='<Required>file directory of dataset', required=True)

    parser.add_argument('--encoder_type', type=str,
                        help='<Required> name of encoder(ResNet, MLP)', required=True)
    parser.add_argument('--dir2save', type=str,
                        help='<Required> name of directory to save computed data', required=True)
    parser.add_argument('--feature_list', type=int, nargs='+',
                        help='feature list to pre train', default=[])

    parser.add_argument('--seed', type=int, help='random seed used in training process', default=0)
    parser.add_argument('--batch_size', type=int, help='batch_size when train', default=128)
    parser.add_argument('--hidden_dim', type=int, help='hidden_dim of network', default=128)
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help='shuffle the dataset?')

    parser.add_argument('--cuda', type=int, help='<Required> cuda to train?', required=True)

    args = parser.parse_args()
    dataset_dir = args.dataset_dir

    save_file_dir = args.dir2save
    try_mkdir(save_file_dir)

    feature_list = args.feature_list
    encoder_type = args.encoder_type

    # Hyper parameters
    seed = args.seed
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    shuffle = args.shuffle

    # cuda
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.cuda))
    else:
        device = torch.device("cpu")


    # save model to excel file
    feature_str = str(feature_list)
    cur_feature_dir = os.path.join(save_file_dir, feature_str)
    try_mkdir(cur_feature_dir)

    encoder_file = os.path.join(cur_feature_dir, 'encoder.pt')
    head_file = os.path.join(cur_feature_dir, 'head.pt')

    if os.path.exists(encoder_file) and os.path.exists(head_file):
        print(f'{feature_str} model has been trained')
        return
    setRandomSeed(seed)
    loader_container = LoaderContainer(dataset_dir, batch_size=batch_size, shuffle=shuffle, scaler_type='StandardScaler')
    setRandomSeed(seed)
    workflow = WorkFlow(loader_container, hidden_dim, encoder_type)
    setRandomSeed(seed)
    encoder, heads, epochs_train, train_loss_list, val_loss_list \
        = workflow.MOTIVATION_one_phase_train(feature_list, device=device)
    # save pretrained model
    for feature_head, feature_index in zip(heads[:-1], feature_list):
        torch.save(feature_head, os.path.join(cur_feature_dir, str(feature_index) + '-feature_head.pt'))

    save_image(epochs_train, train_loss_list, val_loss_list, os.path.join(cur_feature_dir, 'train.svg'))
    torch.save(encoder, encoder_file)
    torch.save(heads[-1], head_file)

    # save epochs info
    epochs_num = np.array([epochs_train])
    epoch_info_file_name = os.path.join(cur_feature_dir, 'epoch_info.csv')
    np.savetxt(fname=epoch_info_file_name, X=epochs_num, delimiter=',')


if __name__ == '__main__':
    Motivation_train()
