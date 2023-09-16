import torch
import random
import numpy as np
import os
from WorkFlow import WorkFlow
from LoaderContainer import LoaderContainer
import pandas as pd
from itertools import combinations
from Analyse import Analyse
from TrainEvalFunc import eval
from matplotlib import pyplot as plt

def save_image(epochs, train_losses, val_losses, file_name):
    if epochs == 0:
        return
    fig, ax = plt.subplots()
    x = np.arange(0, len(train_losses))


    ax.plot(x, train_losses, label='train')
    ax.plot(x, val_losses, label='val')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    plt.legend()
    plt.savefig(file_name, bbox_inches='tight', dpi=300, format='svg')
    plt.close(fig)
def setRandomSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def try_mkdir(dir2make):
    try:
        os.mkdir(dir2make)
    except FileExistsError:
        print(f"相对路径目录'{dir2make}'已经存在。")
    except Exception as e:
        print(f"创建相对路径目录'{dir2make}'时发生错误：{e}")


def eval_from_dir(directory: str, loader_container: LoaderContainer):
    # cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test = []
    val = []
    epochs_num_pre_train = []
    epochs_num_train = []
    feature_list_list = []
    # encoder.pt and head.pt were saved by train_save_eval_models.py
    contents = os.listdir(directory)
    for item in contents:
        if os.path.isdir(directory + item):
            files = os.listdir(directory + item)
            if 'encoder.pt' not in files or 'head.pt' not in files :
                break
            print(directory + item)

            if len(item) > 2:
                feature_list = [int(feature) for feature in item[1:-1].split(',')]
            else:
                feature_list = []

            feature_list_list.append(feature_list)
            trained_encoder = torch.load(directory + item + '/encoder.pt')
            trained_head = torch.load(directory + item + '/head.pt')
            test_data, val_data = eval(loader_container, trained_encoder, trained_head, device)
            # epoch_info
            epoch_info_file_name = 'epoch_info.csv'
            epochs_info = np.loadtxt(directory + item + '/' + epoch_info_file_name, delimiter=',')
            print('------')
            print(epochs_info)
            epochs_num_pre_train.append(epochs_info[0])
            epochs_num_train.append(epochs_info[1])
            test.append(test_data)
            val.append(val_data)
    return test, val, epochs_num_pre_train, epochs_num_train, feature_list_list


def train_and_save_one_model(dataset_dir, dir_name2save, encoder_type, feature_list, seed, batch_size, hidden_dim, shuffle):
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='<Required>file directory of dataset', required=True)

    parser.add_argument('--encoder_type', type=str, help='<Required> name of encoder(ResNet, MLP)', required=True)
    parser.add_argument('--dir_name2save', type=str,
                        help='<Required> name of directory to save computed data', required=True)
    parser.add_argument('--feature_list', type=int, nargs='+',
                        help='feature list to pre train', default=[])

    parser.add_argument('--seed', type=int, help='seed when train', default=0)
    parser.add_argument('--batch_size', type=int, help='batch_size when train', default=128)
    parser.add_argument('--hidden_dim', type=int, help='hidden_dim of network', default=128)
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help='shuffle the dataset?')

    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    dir_name2save = args.dir_name2save
    save_file_dir = dataset_dir + 'dir_name2save/'
    try_mkdir(save_file_dir)

    feature_list = args.feature_list
    encoder_type = args.encoder_type

    # Hyper parameters
    seed = args.seed
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    shuffle = args.shuffle
    '''
    # cuda
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    save_file_dir = dataset_dir + dir_name2save + '/'
    try_mkdir(save_file_dir)

    # save model to excel file
    feature_str = str(feature_list)
    cur_feature_dir = save_file_dir + feature_str + '/'
    try_mkdir(cur_feature_dir)
    encoder_file = cur_feature_dir + 'encoder' + '.pt'
    head_file = cur_feature_dir + 'head' + '.pt'

    if os.path.exists(encoder_file) and os.path.exists(head_file):
        return
    loader_container = LoaderContainer(dataset_dir, batch_size, shuffle)
    analysis = Analyse(loader_container)
    setRandomSeed(seed)
    workflow = WorkFlow(loader_container, hidden_dim, encoder_type)
    setRandomSeed(seed)
    encoder, feature_heads, epochs_pre_train, pre_train_loss_list, pre_val_loss_list \
        = workflow.pre_train(feature_list, device=device)
    encoder, head, epochs_train, train_loss_list, val_loss_list = workflow.train(encoder, device=device)
    test_metric, val_metric = workflow.eval(encoder, head, device=device)
    save_image(epochs_pre_train, pre_train_loss_list, pre_val_loss_list, cur_feature_dir + 'pre_train.svg')
    save_image(epochs_train, train_loss_list, val_loss_list, cur_feature_dir + 'train.svg')
    torch.save(encoder, encoder_file)
    torch.save(head, head_file)
    for feature_head, feature_index in zip(feature_heads, feature_list):
        torch.save(feature_head, cur_feature_dir + str(feature_index) + '-feature_head' + '.pt')
    # save epochs info
    epochs_info = np.array([epochs_pre_train, epochs_train])
    epoch_info_file_name = cur_feature_dir + 'epoch_info.csv'
    np.savetxt(fname=epoch_info_file_name, X=epochs_info, delimiter=',')
