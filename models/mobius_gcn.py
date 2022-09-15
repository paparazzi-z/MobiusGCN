from __future__ import absolute_import

import torch.nn as nn
from models.mobius_graph_conv import MobiusGraphConv

from functools import reduce


class _GraphConv(nn.Module):
    def __init__(self, eigenVal, eigenVec, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = MobiusGraphConv(input_dim, output_dim, eigenVal, eigenVec).float()
#         self.bn = nn.BatchNorm1d(output_dim).float()
#         self.relu = nn.ReLU().float()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x)
#         x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

#         x = self.relu(x)
        return x


class MobiusGCN(nn.Module):
    def __init__(self, eigenVal, eigenVec, hid_dim, num_layers, coords_dim=(2, 3), p_dropout=None):
        super(MobiusGCN, self).__init__()

        _gconv_input = [_GraphConv(eigenVal, eigenVec, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        
        _gconv_layers = []
        for i in range(num_layers):
            _gconv_layers.append(_GraphConv(eigenVal, eigenVec, hid_dim, hid_dim, p_dropout=p_dropout))
            
        _gconv_output = [_GraphConv(eigenVal, eigenVec, hid_dim, coords_dim[1], p_dropout=p_dropout)]


        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = nn.Sequential(*_gconv_output)

    def forward(self, x):
        out = self.gconv_input(x)
#         print('out1')
#         print(out)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
#         print('out2')
#         print(out)
        return out
