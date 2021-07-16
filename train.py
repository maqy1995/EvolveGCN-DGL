# -*- coding: utf-8 -*-
"""
@Author: maqy
@Time: 2021/6/28
@Description: 
"""
import argparse

import dgl
import torch
import torch.nn.functional as F

from dataset import EllipticDataset
from model import EvolveGCNO, EvolveGCNH
from utils import Measure


# TODO refactor code, especially about print info and the way to eval.
def train(args, device):
    elliptic_dataset = EllipticDataset(raw_dir=args.raw_dir,
                                       processed_dir=args.processed_dir,
                                       self_loop=True,
                                       reverse_edge=True)

    g, node_mask_by_time = elliptic_dataset.process()
    num_classes = elliptic_dataset.num_classes

    cached_subgraph = []
    cached_valid_node_mask = []
    for i in range(len(node_mask_by_time)):
        # we add self loop edge when we construct full graph, not here
        node_subgraph = dgl.node_subgraph(graph=g, nodes=node_mask_by_time[i])
        cached_subgraph.append(node_subgraph.to(device))
        valid_node_mask = node_subgraph.ndata['label'] >= 0
        cached_valid_node_mask.append(valid_node_mask)

    if args.model == 'EvolveGCN-O':
        model = EvolveGCNO(in_feats=int(g.ndata['feat'].shape[1]),
                           n_hidden=args.n_hidden,
                           num_layers=args.n_layers)
    elif args.model == 'EvolveGCN-H':
        model = EvolveGCNH(in_feats=int(g.ndata['feat'].shape[1]),
                           num_layers=args.n_layers)
    else:
        return NotImplementedError('Unsupported model {}'.format(args.model))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # split train, valid, test(0-30,31-35,36-48)
    time_window_size = args.n_hist_steps
    loss_class_weight = [float(w) for w in args.loss_class_weight.split(',')]
    loss_class_weight = torch.Tensor(loss_class_weight).to(device)

    train_measure = Measure(num_classes=num_classes, target_class=args.eval_class_id)
    valid_measure = Measure(num_classes=num_classes, target_class=args.eval_class_id)
    test_measure = Measure(num_classes=num_classes, target_class=args.eval_class_id)
    # TODO data split from dataset.py
    train_max_index = 30
    valid_max_index = 35
    test_max_index = 48
    for epoch in range(args.num_epochs):
        model.train()
        for i in range(time_window_size, train_max_index + 1):
            g_list = cached_subgraph[i - time_window_size:i + 1]
            predictions = model(g_list)
            # get predictions which has label
            predictions = predictions[cached_valid_node_mask[i]]
            labels = cached_subgraph[i].ndata['label'][cached_valid_node_mask[i]].long()
            # loss = F.cross_entropy(predictions, labels, weight=torch.Tensor([0.35, 0.65]))
            loss = F.cross_entropy(predictions, labels, weight=loss_class_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_measure.append_measures(predictions, labels)

        # get each epoch measure during training.
        cl_precision, cl_recall, cl_f1 = train_measure.get_total_measure()
        train_measure.update_best_f1(cl_f1, epoch)
        # reset measures for next epoch
        train_measure.reset_info()

        print("Train Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
              .format(epoch, args.eval_class_id, cl_precision, cl_recall, cl_f1))

        # eval
        model.eval()
        for i in range(train_max_index + 1, valid_max_index + 1):
            g_list = cached_subgraph[i - time_window_size:i + 1]
            predictions = model(g_list)
            # get node predictions which has label
            predictions = predictions[cached_valid_node_mask[i]]
            labels = cached_subgraph[i].ndata['label'][cached_valid_node_mask[i]].long()

            valid_measure.append_measures(predictions, labels)

        # get each epoch measure during eval.
        cl_precision, cl_recall, cl_f1 = valid_measure.get_total_measure()
        valid_measure.update_best_f1(cl_f1, epoch)
        # reset measures for next epoch
        valid_measure.reset_info()

        print("Eval Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
              .format(epoch, args.eval_class_id, cl_precision, cl_recall, cl_f1))

        # early stop
        if epoch - valid_measure.target_best_f1_epoch > args.patience:
            print("Best eval Epoch {}, Cur Epoch {}".format(valid_measure.target_best_f1_epoch, epoch))
            break
        # if cur valid f1 score is best, do test
        if epoch == valid_measure.target_best_f1_epoch:
            print("###################Epoch {} Test###################".format(epoch))
            for i in range(valid_max_index + 1, test_max_index + 1):
                g_list = cached_subgraph[i - time_window_size:i + 1]
                predictions = model(g_list)
                # get predictions which has label
                predictions = predictions[cached_valid_node_mask[i]]
                labels = cached_subgraph[i].ndata['label'][cached_valid_node_mask[i]].long()

                test_measure.append_measures(predictions, labels)

            # we get each subgraph measure when testing to match fig 4 in EvolveGCN paper.
            cl_precisions, cl_recalls, cl_f1s = test_measure.get_each_timestamp_measure()
            for index, (sub_p, sub_r, sub_f1) in enumerate(zip(cl_precisions, cl_recalls, cl_f1s)):
                print("  Test | Time {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
                      .format(valid_max_index + index + 2, sub_p, sub_r, sub_f1))

            # get each epoch measure during test.
            cl_precision, cl_recall, cl_f1 = test_measure.get_total_measure()
            test_measure.update_best_f1(cl_f1, epoch)
            # reset measures for next epoch
            test_measure.reset_info()

            print("  Test | Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
                  .format(epoch, args.eval_class_id, cl_precision, cl_recall, cl_f1))

    print("Best test f1 is {}, in Epoch {}"
          .format(test_measure.target_best_f1, test_measure.target_best_f1_epoch))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("EvolveGCN")
    argparser.add_argument('--model', type=str, default='EvolveGCN-O',
                           help='we can choose EvolveGCN-O or EvolveGCN-H,'
                                'but the EvolveGCN-H performance in Elliptic dataset is bad.')
    argparser.add_argument('--raw-dir', type=str,
                           default='/home/maqy/gnn2021/gnn/EvolveGCN-master/data/elliptic/elliptic_bitcoin_dataset/',
                           help="the raw data dir after unzip download from kaggle, which contains 3 csv file")
    argparser.add_argument('--processed-dir', type=str,
                           default='/home/maqy/gnn2021/gnn/EvolveGCN-master/data/elliptic_dgl/',
                           help="the dir store processed raw data")
    argparser.add_argument('--gpu', type=int, default=1,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=500)
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
    argparser.add_argument('--patience', type=int, default=100,
                           help="patience for early stopping")

    args = argparser.parse_args()

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    import time

    start_time = time.perf_counter()
    train(args, device)
    print("train time is: {}".format(time.perf_counter() - start_time))
