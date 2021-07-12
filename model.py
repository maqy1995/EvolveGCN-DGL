# -*- coding: utf-8 -*-
"""
@Author: maqy
@Time: 2021/6/25
@Description: 
"""

import torch
import torch.nn as nn
from torch.nn import LSTM
from torch.nn import GRU
from dgl.nn.pytorch import GraphConv
from torch.nn.parameter import Parameter


# class EvolveGCNH(nn.Module):
#     def __init__(self, in_feats=166, n_hidden=76, num_layers=2, n_classes=2):
#         super(EvolveGCNH, self).__init__()
#         self.num_layers = num_layers
#         self.recurrent_layers = nn.ModuleList()
#         self.gnn_convs = nn.ModuleList()
#         self.gcn_weights_list = nn.ParameterList()
#
#         # TODO according to the code of author, I think we need to write LSTM manually.
#         # Here we use torch.nn.GRU directly, like pyg_temporal.
#         # see github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/evolvegcnh.py
#         self.recurrent_layers.append(GRU(input_size=n_hidden, hidden_size=n_hidden))
#         self.gcn_weights_list.append(Parameter(torch.Tensor(in_feats, n_hidden)))
#         self.gnn_convs.append(
#             GraphConv(in_feats=in_feats, out_feats=n_hidden, bias=False, activation=nn.RReLU(), weight=False))
#         for _ in range(num_layers - 1):
#             self.recurrent_layers.append(GRU(input_size=n_hidden, hidden_size=n_hidden))
#             self.gcn_weights_list.append(Parameter(torch.Tensor(n_hidden, n_hidden)))
#             self.gnn_convs.append(
#                 GraphConv(in_feats=n_hidden, out_feats=n_hidden, bias=False, activation=nn.RReLU(), weight=False))
#
#         # 307 from official parameters_elliptic_egcn_o.yaml
#         self.mlp = nn.Sequential(nn.Linear(n_hidden, 307),
#                                  nn.ReLU(),
#                                  nn.Linear(307, n_classes))
#         self.reset_params()
#
#     def reset_params(self):
#         for layer in self.recurrent_layers:
#             layer.reset_parameters()
#         for layer in self.gcn_weights_list:
#             torch.nn.init.xavier_normal_(layer)
#         for layer in self.gnn_convs:
#             layer.reset_parameters()
#
#     def forward(self, g_list):
#         feature_list = []
#         for g in g_list:
#             feature_list.append(g.ndata['feat'])
#         for i in range(self.num_layers):
#             for j, g in enumerate(g_list):
#                 # Attention: I try to use the below code to set gcn.weight(similar to pyG_temporal),
#                 # but it doesn't work. And I make a demo try to get the difference. see test_parameter.py
#                 # ====================================================
#                 # W = self.gnn_convs[i].weight[None, :, :]
#                 # W, _ = self.recurrent_layers[i](W)
#                 # self.gnn_convs[i].weight = nn.Parameter(W.squeeze())
#                 # ====================================================
#
#                 W = self.gcn_weights_list[i][None, :, :]
#                 W, _ = self.recurrent_layers[i](W)
#                 feature_list[j] = self.gnn_convs[i](g, feature_list[j], weight=W.squeeze())
#         return self.mlp(feature_list[-1])


# TODO pay attention to self loop etc.
class EvolveGCNO(nn.Module):
    def __init__(self, in_feats, n_hidden, num_layers=2, n_classes=2):
        super(EvolveGCNO, self).__init__()
        self.num_layers = num_layers
        self.recurrent_layers = nn.ModuleList()
        self.gnn_convs = nn.ModuleList()
        self.gcn_weights_list = nn.ParameterList()

        # TODO according to the code of author, I think we need to write LSTM manually.
        # see: https://github.com/IBM/EvolveGCN/blob/90869062bbc98d56935e3d92e1d9b1b4c25be593/egcn_o.py#L81
        # But in the paper, the author said 'In other words, one uses the same GRU to process each
        # column of the GCN weight matrix.', It makes me think it's okay to use LSTM/GRU directly.
        # Here we use torch.nn.LSTM directly, like pyg_temporal.
        # see github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/evolvegcno.py
        self.recurrent_layers.append(LSTM(input_size=n_hidden, hidden_size=n_hidden))
        self.gcn_weights_list.append(Parameter(torch.Tensor(in_feats, n_hidden)))
        self.gnn_convs.append(
            GraphConv(in_feats=in_feats, out_feats=n_hidden, bias=False, activation=nn.RReLU(), weight=False))
        for _ in range(num_layers - 1):
            self.recurrent_layers.append(LSTM(input_size=n_hidden, hidden_size=n_hidden))
            self.gcn_weights_list.append(Parameter(torch.Tensor(n_hidden, n_hidden)))
            self.gnn_convs.append(
                GraphConv(in_feats=n_hidden, out_feats=n_hidden, bias=False, activation=nn.RReLU(), weight=False))

        # 307 from official parameters_elliptic_egcn_o.yaml
        self.mlp = nn.Sequential(nn.Linear(n_hidden, 307),
                                 nn.ReLU(),
                                 nn.Linear(307, n_classes))
        self.reset_params()

    def reset_params(self):
        for layer in self.recurrent_layers:
            layer.reset_parameters()
        for layer in self.gcn_weights_list:
            torch.nn.init.xavier_normal_(layer)
        for layer in self.gnn_convs:
            layer.reset_parameters()

    def forward(self, g_list):
        feature_list = []
        for g in g_list:
            feature_list.append(g.ndata['feat'])
        for i in range(self.num_layers):
            for j, g in enumerate(g_list):
                # Attention: I try to use the below code to set gcn.weight(similar to pyG_temporal),
                # but it doesn't work. And I make a demo try to get the difference. see test_parameter.py
                # ====================================================
                # W = self.gnn_convs[i].weight[None, :, :]
                # W, _ = self.recurrent_layers[i](W)
                # self.gnn_convs[i].weight = nn.Parameter(W.squeeze())
                # ====================================================
                W = self.gcn_weights_list[i][None, :, :]
                # Remove the following line of code, it will become `GCN`.
                W, _ = self.recurrent_layers[i](W)
                feature_list[j] = self.gnn_convs[i](g, feature_list[j], weight=W.squeeze())
        return self.mlp(feature_list[-1])
