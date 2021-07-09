# -*- coding: utf-8 -*-
"""
@Author: maqy
@Time: 2021/6/28
@Description: 
"""
import argparse
import random

import dgl
import torch
import torch.nn.functional as F
import numpy as np

from data_process import process_data
from model import EvolveGCNO
from utils import get_accuracy, eval_predicitions, calc_eval_measures_per_class


def train(args, device):
    num_classes = 2
    home_path = "/home/maqy/gnn2021/gnn/EvolveGCN-master/data/elliptic_bitcoin_dataset_cont"
    # TODO id_label_mask可能也用不到，直接可以通过g.ndata['label']来判断？ 后续整合到dataset.py中
    g, node_mask_by_time, id_label_mask = process_data(home_path, add_reverse_edge=True)

    cached_subgraph = []
    cached_valid_node_mask = []
    best_eval_f1 = 0
    for i in range(len(node_mask_by_time)):
        node_subgraph = dgl.node_subgraph(graph=g, nodes=node_mask_by_time[i])
        # TODO: should we add self loop? or do it in dataset?
        node_subgraph = dgl.remove_self_loop(node_subgraph)
        node_subgraph = dgl.add_self_loop(node_subgraph)
        cached_subgraph.append(node_subgraph.to(device))
        valid_node_mask = node_subgraph.ndata['label'] >= 0
        cached_valid_node_mask.append(valid_node_mask)

    model = EvolveGCNO(in_feats=int(g.ndata['feat'].shape[1]),
                       n_hidden=args.n_hidden,
                       num_layers=args.n_layers)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # split train, valid, test(0-30,31-35,36-48)
    time_window_size = args.n_hist_steps
    loss_class_weight = [float(w) for w in args.loss_class_weight.split(',')]
    loss_class_weight = torch.Tensor(loss_class_weight).to(device)
    # TODO data split from dataset.py
    train_max_index = 30
    valid_max_index = 35
    test_max_index = 48
    for epoch in range(args.num_epochs):
        model.train()
        total_true_positives = {}
        total_false_negatives = {}
        total_false_positives = {}
        for i in range(num_classes):
            total_true_positives[i] = 0
            total_false_negatives[i] = 0
            total_false_positives[i] = 0
        for i in range(time_window_size, train_max_index + 1):
            g_list = cached_subgraph[i - time_window_size:i + 1]
            predictions = model(g_list)
            # get predictions which has label
            predictions = predictions[cached_valid_node_mask[i]]
            labels = cached_subgraph[i].ndata['label'][cached_valid_node_mask[i]].long()
            # TODO set weight for each class
            # loss = F.cross_entropy(predictions, labels, weight=torch.Tensor([0.35, 0.65]))
            loss = F.cross_entropy(predictions, labels, weight=loss_class_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO num_classes
            error, true_positives, false_negatives, false_positives = eval_predicitions(predictions, labels,
                                                                                        num_classes=num_classes)
            batch_precision, batch_recall, batch_f1 = calc_eval_measures_per_class(true_positives,
                                                                                   false_negatives,
                                                                                   false_positives,
                                                                                   class_id=args.eval_class_id)
            print("batch {} train class 1 | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
                  .format(i + 1, batch_precision, batch_recall, batch_f1))

            for j in range(num_classes):
                total_true_positives[j] += true_positives[j]
                total_false_negatives[j] += false_negatives[j]
                total_false_positives[j] += false_positives[j]

        cl_precision, cl_recall, cl_f1 = calc_eval_measures_per_class(total_true_positives,
                                                                      total_false_negatives,
                                                                      total_false_positives,
                                                                      class_id=args.eval_class_id)

        print("Epoch {} train class 1 | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
              .format(epoch, cl_precision, cl_recall, cl_f1))

        model.eval()
        total_true_positives = {}
        total_false_negatives = {}
        total_false_positives = {}
        for i in range(num_classes):
            total_true_positives[i] = 0
            total_false_negatives[i] = 0
            total_false_positives[i] = 0
        for i in range(train_max_index + 1, valid_max_index + 1):
            g_list = cached_subgraph[i - time_window_size:i + 1]
            predictions = model(g_list)
            # get node predictions which has label
            predictions = predictions[cached_valid_node_mask[i]]
            labels = cached_subgraph[i].ndata['label'][cached_valid_node_mask[i]].long()
            # get accuracy or f1 score
            # TODO get accuracy
            error, true_positives, false_negatives, false_positives = eval_predicitions(predictions, labels,
                                                                                        num_classes=num_classes)
            batch_precision, batch_recall, batch_f1 = calc_eval_measures_per_class(true_positives,
                                                                                   false_negatives,
                                                                                   false_positives,
                                                                                   class_id=args.eval_class_id)
            print("batch {} eval class 1 | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
                  .format(i + 1, batch_precision, batch_recall, batch_f1))
            for j in range(num_classes):
                total_true_positives[j] += true_positives[j]
                total_false_negatives[j] += false_negatives[j]
                total_false_positives[j] += false_positives[j]

        cl_precision, cl_recall, cl_f1 = calc_eval_measures_per_class(total_true_positives,
                                                                      total_false_negatives,
                                                                      total_false_positives,
                                                                      class_id=args.eval_class_id)
        print("Epoch {} eval class 1 | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
              .format(epoch, cl_precision, cl_recall, cl_f1))

        # test
        # model.eval()
        if cl_f1 > best_eval_f1:
            best_eval_f1 = cl_f1
            print("best eval f1 is {:.4f}".format(best_eval_f1))
            total_true_positives = {}
            total_false_negatives = {}
            total_false_positives = {}
            for i in range(num_classes):
                total_true_positives[i] = 0
                total_false_negatives[i] = 0
                total_false_positives[i] = 0
            for i in range(valid_max_index + 1, test_max_index + 1):
                g_list = cached_subgraph[i - time_window_size:i + 1]
                predictions = model(g_list)
                # get predictions which has label
                predictions = predictions[cached_valid_node_mask[i]]
                labels = cached_subgraph[i].ndata['label'][cached_valid_node_mask[i]].long()

                error, true_positives, false_negatives, false_positives = eval_predicitions(predictions, labels,
                                                                                            num_classes=num_classes)
                batch_precision, batch_recall, batch_f1 = calc_eval_measures_per_class(true_positives,
                                                                                       false_negatives,
                                                                                       false_positives,
                                                                                       class_id=args.eval_class_id)
                print("batch {} test class 1 | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
                      .format(i + 1, batch_precision, batch_recall, batch_f1))
                for j in range(num_classes):
                    total_true_positives[j] += true_positives[j]
                    total_false_negatives[j] += false_negatives[j]
                    total_false_positives[j] += false_positives[j]

            cl_precision, cl_recall, cl_f1 = calc_eval_measures_per_class(total_true_positives,
                                                                          total_false_negatives,
                                                                          total_false_positives,
                                                                          class_id=args.eval_class_id)
            print("best Epoch {} test class 1 | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
                  .format(epoch, cl_precision, cl_recall, cl_f1))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("EvolveGCN")
    argparser.add_argument('--dataset', type=str, default='Elliptic')
    argparser.add_argument('--gpu', type=int, default=-1,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--n-hidden', type=int, default=256)
    argparser.add_argument('--n-layers', type=int, default=2)
    argparser.add_argument('--n-hist-steps', type=int, default=5,
                           help="number of previous steps used for prediction."
                                "If it is set to 5, it means that the first batch,"
                                "we use historical data of 0-4 to predict the data of time 5")
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--loss-class-weight', type=str, default='0.35,0.65')
    argparser.add_argument('--eval-class-id', type=int, default=1,
                           help="class type to eval. In Elliptic, the id 1(illicit) is the main interest")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    import time
    start_time = time.perf_counter()
    train(args, device)
    print("train time is: {}".format(time.perf_counter() - start_time))
