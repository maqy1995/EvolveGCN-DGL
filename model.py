# -*- coding: utf-8 -*-
"""
@Author: maqy
@Time: 2021/6/25
@Description:
# TODO the performance is poor than official when use the nn.GRU/LSTM directly,
#  we need to implement GRU/LSTM manually.
"""
import math

import torch
import torch.nn as nn
from torch.nn import LSTM
from torch.nn import GRU
from dgl.nn.pytorch import GraphConv
from torch.nn.parameter import Parameter


def pad_with_last_val(vect, k):
    pad = torch.ones(k - vect.size(0),
                     dtype=torch.long,
                     device=vect.device) * vect[-1]
    vect = torch.cat([vect, pad])
    return vect


class TopK(torch.nn.Module):
    """
    similar to official `egcn_h.py`. we only consider the node in a timestamp based subgraph,
    so we need to pay attention to K should be less than min nodes in all subgraph.
    """

    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_parameters(self.scorer)

        self.k = k

    def reset_parameters(self, t):
        # Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()

        vals, topk_indices = scores.view(-1).topk(self.k)

        # if topk_indices.size(0) < self.k:
        #     topk_indices = pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
                isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        # we need to transpose the output
        return out.t()


class EvolveGCNH(nn.Module):
    def __init__(self, in_feats=166, n_hidden=76, num_layers=2, n_classes=2, classifier_hidden=510):
        super(EvolveGCNH, self).__init__()
        self.num_layers = num_layers
        self.pooling_layers = nn.ModuleList()
        self.recurrent_layers = nn.ModuleList()
        self.gnn_convs = nn.ModuleList()
        self.gcn_weights_list = nn.ParameterList()

        self.pooling_layers.append(TopK(in_feats, n_hidden))
        # according to the code of author, we may need to implement GRU manually.
        # Here we use torch.nn.GRU directly, like pyg_temporal.
        # see github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/evolvegcnh.py
        self.recurrent_layers.append(GRU(input_size=n_hidden, hidden_size=n_hidden))
        self.gcn_weights_list.append(Parameter(torch.Tensor(in_feats, n_hidden)))
        self.gnn_convs.append(
            GraphConv(in_feats=in_feats, out_feats=n_hidden, bias=False, activation=nn.RReLU(), weight=False))
        for _ in range(num_layers - 1):
            self.pooling_layers.append(TopK(n_hidden, n_hidden))
            self.recurrent_layers.append(GRU(input_size=n_hidden, hidden_size=n_hidden))
            self.gcn_weights_list.append(Parameter(torch.Tensor(n_hidden, n_hidden)))
            self.gnn_convs.append(
                GraphConv(in_feats=n_hidden, out_feats=n_hidden, bias=False, activation=nn.RReLU(), weight=False))

        # 510 from official parameters_elliptic_egcn_h.yaml
        self.mlp = nn.Sequential(nn.Linear(n_hidden, classifier_hidden),
                                 nn.ReLU(),
                                 nn.Linear(classifier_hidden, n_classes))
        self.reset_params()

    def reset_params(self):
        for layer in self.gcn_weights_list:
            torch.nn.init.xavier_normal_(layer)

    def forward(self, g_list):
        feature_list = []
        for g in g_list:
            feature_list.append(g.ndata['feat'])
        for i in range(self.num_layers):
            W = self.gcn_weights_list[i][None, :, :]
            for j, g in enumerate(g_list):
                # Attention: I try to use the below code to set gcn.weight(similar to pyG_temporal),
                # but it doesn't work. And I make a demo try to get the difference. see test_parameter.py
                # ====================================================
                # W = self.gnn_convs[i].weight[None, :, :]
                # W, _ = self.recurrent_layers[i](W)
                # self.gnn_convs[i].weight = nn.Parameter(W.squeeze())
                # ====================================================
                X_tilde = self.pooling_layers[i](feature_list[j])
                X_tilde = X_tilde[None, :, :]
                _, W = self.recurrent_layers[i](X_tilde, W)
                feature_list[j] = self.gnn_convs[i](g, feature_list[j], weight=W.squeeze())
        return self.mlp(feature_list[-1])


class EvolveGCNO(nn.Module):
    def __init__(self, in_feats, n_hidden, num_layers=2, n_classes=2, classifier_hidden=307):
        super(EvolveGCNO, self).__init__()
        self.num_layers = num_layers
        self.recurrent_layers = nn.ModuleList()
        self.gnn_convs = nn.ModuleList()
        self.gcn_weights_list = nn.ParameterList()

        # According to the code of author, we may need to write LSTM manually.
        # PS: the code by official seems not use LSTM to implement EvolveGCN-O, they use GRU directly.
        # See: https://github.com/IBM/EvolveGCN/blob/90869062bbc98d56935e3d92e1d9b1b4c25be593/egcn_o.py#L81
        # But in the paper, the author said 'In other words, one uses the same GRU to process each
        # column of the GCN weight matrix.', It makes me think it's okay to use LSTM/GRU directly.
        # ~~Here we use torch.nn.LSTM directly, like pyg_temporal.~~
        # update: use LSTM directly will reduce f1 score(about 0.1).
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
        self.mlp = nn.Sequential(nn.Linear(n_hidden, classifier_hidden),
                                 nn.ReLU(),
                                 nn.Linear(classifier_hidden, n_classes))
        self.reset_params()

    def reset_params(self):
        for layer in self.gcn_weights_list:
            torch.nn.init.xavier_normal_(layer)

    def forward(self, g_list):
        feature_list = []
        for g in g_list:
            feature_list.append(g.ndata['feat'])
        for i in range(self.num_layers):
            W = self.gcn_weights_list[i][None, :, :]
            for j, g in enumerate(g_list):
                # Attention: I try to use the below code to set gcn.weight(similar to pyG_temporal),
                # but it doesn't work. And I make a demo try to get the difference. see `test_parameter.py`
                # ====================================================
                # W = self.gnn_convs[i].weight[None, :, :]
                # W, _ = self.recurrent_layers[i](W)
                # self.gnn_convs[i].weight = nn.Parameter(W.squeeze())
                # ====================================================

                # Remove the following line of code, it will become `GCN`.
                W, _ = self.recurrent_layers[i](W)
                feature_list[j] = self.gnn_convs[i](g, feature_list[j], weight=W.squeeze())
        return self.mlp(feature_list[-1])
