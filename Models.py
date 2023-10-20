from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import typing as ty
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

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
    def __init__(
            self,
            *,
            input_dim: int,
            hid_dim: int,
            n_layers: int,
            dropout: float):
        super(ResNet, self).__init__()

        self.input_layer = nn.Linear(input_dim, hid_dim)
        layer_num = n_layers
        self.dropout_layer = nn.ModuleList(
            [nn.Dropout(p=dropout) for _ in range(layer_num)]
        )
        self.layers = nn.ModuleList(
            [nn.Linear(hid_dim, hid_dim) for _ in range(layer_num)]
        )

    def forward(self, x):
        hid = self.input_layer(x)

        for i, layer in enumerate(self.layers):
            hid = hid + self.dropout_layer[i](F.leaky_relu(layer(hid)))

        return hid

class ResNet2(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        normalization: str,
        hidden_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()

        def make_normalization():
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](hidden_dim)

        self.main_activation = F.relu  # relu
        self.last_activation = F.relu  # relu
        self.residual_dropout = residual_dropout  # true
        self.hidden_dropout = hidden_dropout

        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(),
                        'linear0': nn.Linear(
                            hidden_dim, hidden_dim
                        ),
                        'linear1': nn.Linear(hidden_dim, hidden_dim),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        # self.head = nn.Linear(d, d_out)

    def forward(self, x) -> Tensor:

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        #   x = self.head(x)
        #   x = x.squeeze(-1)
        return x

class Encoder(nn.Module):
    def __init__(self, model_type, input_num, hidden_dim, num_list, cat_list) -> None:
        super().__init__()

        self.input_num = input_num
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.num_list = num_list
        self.cat_list = cat_list
        self.encoder = None
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
        assert (self.model_type == 'MLP' or self.model_type == 'ResNet' or self.model_type == 'ResNet2')
        if self.model_type == 'MLP':
            self.encoder = MLP(
                self.input_dim, self.hidden_dim
            )
        elif self.model_type == 'ResNet':
            self.encoder = ResNet(
                input_dim=self.input_dim,
                hid_dim=self.hidden_dim,
                n_layers=7,
                dropout=0.2
            )
        elif self.model_type == 'ResNet2':
            self.encoder = ResNet2(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                n_layers=10,
                normalization='batchnorm',
                hidden_dropout=0.3,
                residual_dropout=0.3,
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



