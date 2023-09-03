from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)


class ResNet(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(ResNet, self).__init__()

        self.input_layer = nn.Linear(input_dim, hid_dim)
        layer_num = 7
        self.dropout_layer = nn.ModuleList(
            [nn.Dropout(p=0.2) for _ in range(layer_num)]
        )
        self.layers = nn.ModuleList(
            [nn.Linear(hid_dim, hid_dim) for _ in range(layer_num)]
        )

    def forward(self, x):
        hid = self.input_layer(x)

        for i, layer in enumerate(self.layers):
            hid = hid + self.dropout_layer[i](F.leaky_relu(layer(hid)))

        return hid


class Encoder(nn.Module):
    def __init__(self, model_type, input_num, hidden_dim, num_list, cat_list) -> None:
        super().__init__()

        self.input_num = input_num
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.num_list = num_list
        self.cat_list = cat_list
        self._build_preprocessor()
        self.build_model()

    def _build_preprocessor(self):
        self.preprocessor = nn.ModuleList()
        self.input_dim = 0
        self.embed_size = 4

        for i in range(self.input_num):
            if i in self.num_list:
                self.preprocessor.append(nn.Identity())  # nn.Idensity()用于占位
                self.input_dim += 1
            elif i in self.cat_list:
                self.preprocessor.append(
                    nn.Embedding(
                        # !!! max classes for each category features, here is set to 128, better feature specific
                        128,
                        self.embed_size
                    )
                )
                self.input_dim += self.embed_size

    def _preprocess(self, x):
        outs = []
        for i in range(x.size(1)):
            if isinstance(self.preprocessor[i],
                          nn.Identity):  # isinstance(object,classinfo)判断object是不是classinfo(包含继承关系)
                _out = self.preprocessor[i](x.select(1, i).float())  # numerical
                _out = _out.unsqueeze(dim=-1)
            else:
                _out = self.preprocessor[i](x.select(1, i).long())  # categorical
            outs.append(_out)

        outs = torch.cat(outs, dim=-1)

        return outs

    def build_model(self):
        assert (self.model_type == 'MLP' or self.model_type == 'ResNet')
        if self.model_type == 'MLP':
            self.encoder = MLP(
                self.input_dim, self.hidden_dim
            )
        elif self.model_type == 'ResNet':
            self.encoder = ResNet(
                self.input_dim, self.hidden_dim
            )

    def forward(self, inputs):
        return self.encoder(self._preprocess(inputs))


class Head(nn.Module):
    def __init__(self, hidden_dim, out_dim) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.out = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, inputs):
        return self.out(inputs)



