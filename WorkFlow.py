import torch
import random
import numpy as np
import os
import Models
from LoaderContainer import LoaderContainer, TaskType
from TrainEvalFunc import RMSE, multiclass_accuracy, binclass_accuracy, fit
import pandas as pd


class WorkFlow:
    def __init__(self, fileDir, batch_size, hidden_dim, shuffle):
        self.fileDir = fileDir
        self.loader_container = LoaderContainer(fileDir, shuffle)
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.shuffle = shuffle

    def get_feature_num(self):
        input_num, _1, _2, _3 = self.loader_container.getInfo()
        return input_num

    def pre_train(self, feature_list, device):
        input_num, num_list, cat_list, task_type = self.loader_container.getInfo()
        encoder = Models.Encoder('ResNet', input_num, self.hidden_dim, num_list, cat_list).to(device)
        pre_train_head_list = []
        pre_train_loader_list = []
        pre_val_loader_list = []
        pre_train_target_std_list = []
        pre_train_loss_func_list = []  # nn.MSELoss()
        for feature_index in feature_list:
            pre_train_head, pre_train_loss_func = \
                self.loader_container.getPreTrainHeadAndLossFunc(self.hidden_dim, feature_index)
            pre_train_head_list.append(pre_train_head.to(device))
            pre_train_loss_func_list.append(pre_train_loss_func)

            pre_train_loader, pre_val_loader, pre_train_target_std = self.loader_container.getPreTrainLoader(
                self.batch_size,
                feature_index)
            pre_train_loader_list.append(pre_train_loader)
            pre_val_loader_list.append(pre_val_loader)
            pre_train_target_std_list.append(pre_train_target_std)

        encoder, heads = fit(encoder=encoder, loss_func_list=pre_train_loss_func_list, head_list=pre_train_head_list,
                             train_loader_list=pre_train_loader_list, val_loader_list=pre_val_loader_list,
                             target_std_list=pre_train_target_std_list, device=device)
        return encoder, heads

    def train(self, encoder, device):
        target_head, loss_func = self.loader_container.getTrainHeadAndLossFunc(self.hidden_dim)
        target_head = target_head.to(device)
        target_std = self.loader_container.getTargetStd()
        train_loader = self.loader_container.getTrainLoader(self.batch_size)
        val_data_loader = self.loader_container.getValLoader(self.batch_size)
        encoder, head_list = fit(encoder=encoder, loss_func_list=[loss_func], head_list=[target_head],
                                 train_loader_list=[train_loader],
                                 val_loader_list=[val_data_loader], target_std_list=[target_std], device=device)
        return encoder, head_list[0]

    def eval(self, encoder, head, device):
        test_data_loader = self.loader_container.getTestLoader(self.batch_size)
        val_data_loader = self.loader_container.getValLoader(self.batch_size)
        target_std = self.loader_container.getTargetStd()
        _1, _2, _3, task_type = self.loader_container.getInfo()
        if task_type == TaskType.regression:
            val_rmse = RMSE(data_loader=val_data_loader, encoder=encoder, head=head,
                            target_std=target_std, device=device)
            test_rmse = RMSE(data_loader=test_data_loader, encoder=encoder, head=head,
                             target_std=target_std, device=device)
            print(f'测试集的RMSE为{test_rmse} 验证集的RMSE为{val_rmse}')
            return test_rmse, val_rmse
        elif task_type == TaskType.multiclass:
            test_acc = multiclass_accuracy(data_loader=test_data_loader, encoder=encoder, head=head,
                                           device=device)
            val_acc = multiclass_accuracy(data_loader=val_data_loader, encoder=encoder, head=head,
                                          device=device)
            print(f'测试集准确率为{test_acc} 验证集的准确率为{val_acc}')
            return test_acc, val_acc
        elif task_type == TaskType.binclass:
            test_acc = binclass_accuracy(data_loader=test_data_loader, encoder=encoder, head=head, device=device)
            val_acc = binclass_accuracy(data_loader=val_data_loader, encoder=encoder, head=head, device=device)
            print(f'测试集准确率为{test_acc} 验证集的准确率为{val_acc}')
            return test_acc, val_acc



