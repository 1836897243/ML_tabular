import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt
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