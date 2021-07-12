# -*- coding: utf-8 -*-
"""
@Author: maqy
@Time: 2021/7/9
@Description: utils from official code.
"""
import torch


def eval_predicitions(predictions, true_classes, num_classes):
    predicted_classes = predictions.argmax(dim=1)
    failures = (predicted_classes != true_classes).sum(dtype=torch.float)
    error = failures / predictions.size(0)

    true_positives = {}
    false_negatives = {}
    false_positives = {}

    for cl in range(num_classes):
        cl_indices = true_classes == cl

        pos = predicted_classes == cl
        hits = (predicted_classes[cl_indices] == true_classes[cl_indices])

        tp = hits.sum()
        fn = hits.size(0) - tp
        fp = pos.sum() - tp

        true_positives[cl] = tp
        false_negatives[cl] = fn
        false_positives[cl] = fp
    return error, true_positives, false_negatives, false_positives


def calc_eval_measures_per_class(tp, fn, fp, class_id):
    # ALDO
    if type(tp) is dict:
        tp_sum = tp[class_id].item()
        fn_sum = fn[class_id].item()
        fp_sum = fp[class_id].item()
    else:
        tp_sum = tp.item()
        fn_sum = fn.item()
        fp_sum = fp.item()
    ########
    if tp_sum == 0:
        return 0, 0, 0

    p = tp_sum * 1.0 / (tp_sum + fp_sum)
    r = tp_sum * 1.0 / (tp_sum + fn_sum)
    if (p + r) > 0:
        f1 = 2.0 * (p * r) / (p + r)
    else:
        f1 = 0
    return p, r, f1
