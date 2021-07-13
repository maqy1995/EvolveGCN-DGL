# -*- coding: utf-8 -*-
"""
@Author: maqy
@Time: 2021/7/12
@Description: deprecated, please use dataset.py
"""
import pandas
import numpy

home_path = "/home/maqy/gnn2021/gnn/EvolveGCN-master/data/elliptic/elliptic_bitcoin_dataset/"
id_label = pandas.read_csv(home_path + 'elliptic_txs_classes.csv')
src_dst = pandas.read_csv(home_path + 'elliptic_txs_edgelist.csv')
# elliptic_txs_features.csv has no header, and it has the same order idx with elliptic_txs_classes.csv
id_time_features = pandas.read_csv(home_path + 'elliptic_txs_features.csv', header=None)

# get oldId_newId
oid_nid = id_label.loc[:, ['txId']]
oid_nid = oid_nid.rename(columns={'txId': 'originalId'})
oid_nid.insert(1, 'newId', range(len(oid_nid)))

# map classes unknown,1,2 to -1,1,0 and construct id_label. type 1 means illicit.
id_label = pandas.concat(
    [oid_nid['newId'], id_label['class'].map({'unknown': -1.0, '1': 1.0, '2': 0.0})], axis=1)

# replace originalId in id_time_features to newId.
# Attention: the timestamp in features start at 1; id in id_time_features also has changed.
id_time_features[0] = oid_nid['newId']

# construct originalId2newId dict
oid_nid_dict = oid_nid.set_index(['originalId'])['newId'].to_dict()
# construct newId2timestamp dict
nid_time_dict = id_time_features.set_index([0])[1].to_dict()

# map id in edgelist to newId, and add a timestamp to each edge.
# Attention: From the EvolveGCN official instruction, the timestamp with edgelist start at 0, rather than 1.
# see: github.com/IBM/EvolveGCN/blob/master/elliptic_construction.md
# Here we dose not follow the official instruction, which means timestamp with edgelist also start at 1.
#
# note: in the dataset, src and dst node has the same timestamp.
new_src = src_dst['txId1'].map(oid_nid_dict).rename('newSrc')
new_dst = src_dst['txId2'].map(oid_nid_dict).rename('newDst')
edge_time = new_src.map(nid_time_dict).rename('timestamp')
src_dst_time = pandas.concat([new_src, new_dst, edge_time], axis=1)

# save oid_nid, id_label, id_time_features, src_dst_time to disk. we can convert them to numpy.
# oid_nid: type int.  id_label: type int.   id_time_features: type float.  src_dst_time: type int.
oid_nid = oid_nid.to_numpy(dtype=int)
id_label = id_label.to_numpy(dtype=int)
id_time_features = id_time_features.to_numpy(dtype=float)
src_dst_time = src_dst_time.to_numpy(dtype=int)

path = "/home/maqy/gnn2021/gnn/EvolveGCN-master/data/elliptic_dgl/"
numpy.save(path+'oid_nid.npy', oid_nid)
numpy.save(path+'id_label.npy', id_label)
numpy.save(path+'id_time_features.npy', id_time_features)
numpy.save(path+'src_dst_time.npy', src_dst_time)