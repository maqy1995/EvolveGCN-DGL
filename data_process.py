# -*- coding: utf-8 -*-
"""
@Author: maqy
@Time: 2021/6/21
@Description: deprecated, please use dataset.py
"""
import numpy
import pandas
import dgl
import warnings
import torch

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")


def process_data(home, add_reverse_edge=True):
    id_time_features_path = home + "/elliptic_txs_features.csv"

    id_label_path = home + "/elliptic_txs_classes.csv"

    src_dst_time_path = home + "/elliptic_txs_edgelist_timed.csv"

    id_time_path = home + "/elliptic_txs_nodetime.csv"

    originalID_continusID_path = home + "/elliptic_txs_orig2contiguos.csv"

    id_time_features = pandas.read_csv(id_time_features_path, dtype=float, header=None)

    id_label = pandas.read_csv(id_label_path, dtype=int)

    src_dst_time = pandas.read_csv(src_dst_time_path, dtype=int)

    # 167 numbers
    # create dgl graph
    id_time_features = torch.Tensor(id_time_features.to_numpy(dtype=float))
    id_label = torch.IntTensor(id_label.to_numpy(dtype=int))
    src_dst_time = torch.IntTensor(src_dst_time.to_numpy(dtype=int))

    # path = "/home/maqy/gnn2021/gnn/EvolveGCN-master/data/elliptic_dgl/"
    # id_time_features = torch.Tensor(numpy.load(path+'id_time_features.npy'))
    # id_label = torch.IntTensor(numpy.load(path+'id_label.npy'))
    # src_dst_time = torch.IntTensor(numpy.load(path+'src_dst_time.npy'))
    # label >= 0
    id_label_mask = id_label[:, 1] >= 0

    src = src_dst_time[:, 0]
    dst = src_dst_time[:, 1]
    # id_label[:, 0] is used to add self loop
    if add_reverse_edge:
        g = dgl.graph(data=(torch.cat((src, dst, id_label[:, 0])), torch.cat((dst, src, id_label[:, 0]))),
                      num_nodes=id_label.shape[0])
    else:
        g = dgl.graph(data=(torch.cat((src, id_label[:, 0])), torch.cat((dst, id_label[:, 0]))),
                      num_nodes=id_label.shape[0])

    node_mask_by_time = []
    # edge_mask_by_time = []
    start_time = int(torch.min(id_time_features[:, 1]))
    end_time = int(torch.max(id_time_features[:, 1]))
    # TODO 后续这里要处理一下,不过edge_mask实际应该用不到
    # 注意: src_dst_time中的time从0开始，id_time_features中的time从1开始
    for i in range(start_time, end_time + 1):
        node_mask = id_time_features[:, 1] == i
        # edge_mask = src_dst_time[:, 2] == i - 1
        node_mask_by_time.append(node_mask)
        # edge_mask_by_time.append(edge_mask)

    time_features = id_time_features[:, 1:]
    label = id_label[:, 1]
    g.ndata['label'] = label
    g.ndata['feat'] = time_features

    if add_reverse_edge:
        g.edata['timestamp'] = torch.cat((src_dst_time[:, 2], src_dst_time[:, 2], id_time_features[:, 1].int()))
    else:
        g.edata['timestamp'] = torch.cat(src_dst_time[:, 2], id_time_features[:, 1].int())

    return g, node_mask_by_time, id_label_mask


# TODO get each timed-subgraph by node_mask_by_time[i] * id_label_mask
# 测试通过node采样和edge采样是否是相等的。（即，node都使用时间为1的，edge也使用时间为1的）
def compare_subgraph(g, node_mask_by_time, edge_mask_by_time):
    """
    test if subgraph by node time is equal to subgraph by edge time
    """
    assert len(node_mask_by_time) == len(edge_mask_by_time)

    for i in range(len(node_mask_by_time)):
        node_subgraph = dgl.node_subgraph(graph=g, nodes=node_mask_by_time[i], store_ids=True)
        edge_subgraph = dgl.edge_subgraph(graph=g, edges=edge_mask_by_time[i], store_ids=True)
        if node_subgraph.num_nodes() != edge_subgraph.num_nodes() or \
                node_subgraph.num_edges() != edge_subgraph.num_edges():
            return False
        # nodes or edges may out of order, so we need to sort it first
        node_id1, _ = torch.sort(node_subgraph.ndata[dgl.NID])
        node_id2, _ = torch.sort(edge_subgraph.ndata[dgl.NID])

        edge_id1, _ = torch.sort(node_subgraph.edata[dgl.EID])
        edge_id2, _ = torch.sort(edge_subgraph.edata[dgl.EID])

        if not torch.equal(node_id1, node_id2) or not torch.equal(edge_id1, edge_id2):
            return False

    print('subgraph by edge or node is equal!')
    return True


if __name__ == "__main__":
    home_path = "/home/maqy/gnn2021/gnn/EvolveGCN-master/data/elliptic_bitcoin_dataset_cont"
    # set add_reverse_edge to `False` to run compare_subgraph.
    g, node_mask_by_time, edge_mask_by_time, id_label_mask = process_data(home_path, add_reverse_edge=False)
    node_subgraph = dgl.node_subgraph(graph=g, nodes=node_mask_by_time[0])
