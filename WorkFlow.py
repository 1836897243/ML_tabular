import torch
import random
import numpy as np
import os
import Models
from LoaderContainer import LoaderContainer, TaskType
from TrainEvalFunc import RMSE, multiclass_accuracy, binclass_accuracy, fit
import pandas as pd


class WorkFlow:
    def __init__(self, loader_container, hidden_dim, encoder_type: str):
        self.loader_container = loader_container
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type

    def get_feature_num(self):
        input_num, _1, _2, _3 = self.loader_container.getInfo()
        return input_num

    def pre_train(self, feature_list, device):
        input_num, num_list, cat_list, task_type = self.loader_container.getInfo()
        encoder = Models.Encoder(self.encoder_type, input_num, self.hidden_dim, num_list, cat_list).to(device)
        pre_train_head_list = []
        pre_train_loader_list = []
        pre_val_loader_list = []
        pre_train_loss_func_list = []  # nn.MSELoss()

        all_pre_train_head_list, all_pre_train_loss_func_list = \
            self.loader_container.getAllPreTrainHeadAndLossFuncList(self.hidden_dim)

        for feature_index in feature_list:
            # pre_train_head, pre_train_loss_func = \
            #    self.loader_container.getPreTrainHeadAndLossFunc(self.hidden_dim, feature_index)
            # pre_train_head_list.append(pre_train_head.to(device))
            # pre_train_loss_func_list.append(pre_train_loss_func)

            pre_train_head_list.append(all_pre_train_head_list[feature_index].to(device))
            pre_train_loss_func_list.append(all_pre_train_loss_func_list[feature_index])

            pre_train_loader, pre_val_loader = \
                self.loader_container.getPreTrainLoader(feature_index)
            pre_train_loader_list.append(pre_train_loader)
            pre_val_loader_list.append(pre_val_loader)

        encoder, heads, epochs, train_loss_list, val_loss_list = fit(encoder=encoder, loss_func_list=pre_train_loss_func_list, head_list=pre_train_head_list,
                             train_loader_list=pre_train_loader_list, val_loader_list=pre_val_loader_list,
                             device=device, early_stop=16)
        return encoder, heads, epochs, train_loss_list, val_loss_list

    def train(self, encoder, device):
        target_head, loss_func = self.loader_container.getTrainHeadAndLossFunc(self.hidden_dim)
        target_head = target_head.to(device)
        train_loader = self.loader_container.getTrainLoader()
        val_data_loader = self.loader_container.getValLoader()
        encoder, head_list, epochs, train_loss_list, val_loss_list = fit(encoder=encoder, loss_func_list=[loss_func], head_list=[target_head],
                                 train_loader_list=[train_loader], val_loader_list=[val_data_loader],
                                device=device, early_stop=16)

        return encoder, head_list[0], epochs, train_loss_list, val_loss_list

    def eval(self, encoder, head, device):
        test_data_loader = self.loader_container.getTestLoader()
        val_data_loader = self.loader_container.getValLoader()
        inverse_transform_func = self.loader_container.getInverseTransformFunc()
        _1, _2, _3, task_type = self.loader_container.getInfo()
        if task_type == TaskType.regression:
            val_rmse = RMSE(data_loader=val_data_loader, encoder=encoder, head=head,
                            inverse_transform_func=inverse_transform_func, device=device)
            test_rmse = RMSE(data_loader=test_data_loader, encoder=encoder, head=head,
                             inverse_transform_func=inverse_transform_func, device=device)
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



