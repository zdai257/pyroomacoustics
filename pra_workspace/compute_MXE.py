import librosa
import soundfile
import itertools
from aubio import source, onset
import numpy as np
import math
import random
import os
from os.path import join
import json


# Sort the annotation dictionary into lists by Polyphony attribute, e.g., 'PolyphonyLevel'
def eval_polyphony(ann, sort_attr='PolyphonyLevel'):
    PR_lst = sorted(ann.items(), key=lambda x: x[1][sort_attr], reverse=False)

    attr_lst, gt_count_lst, pred_count_lst = [], [], []

    for item in PR_lst:
        if item[1]['count_num_pred'] is not None:
            attr_lst.append(item[1][sort_attr])

            gt_count_lst.append(item[1]['count'])
            pred_count_lst.append(item[1]['count_num_pred'])

    print("Number of test samples = ", len(attr_lst))
    results = (attr_lst, gt_count_lst, pred_count_lst)
    return results


### Compute MSE, MDE, Accuracy ###
# y_gt: ground truth list
# y_pred: prediction list
# p: tolerance
def get_metric(y_gt, y_pred, p=0):
    mse_sum = 0
    mde_sum = 0
    acc_sum = 0

    if len(y_pred) == 0:
        return None, None, None

    for idx, gt in enumerate(y_gt):
        if p == 0:
            if isinstance(y_pred[idx], list):
                pred_val = np.mean(y_pred[idx])

                acc_sum += sum([item == gt for item in y_pred[idx]]) / len(y_pred[idx])
            else:
                pred_val = y_pred[idx]
                if gt == pred_val:
                    acc_sum += 1
        elif p == 1:
            if isinstance(y_pred[idx], list):
                pred_val = np.mean(y_pred[idx])
                batch_sum = 0
                for item in y_pred[idx]:
                    if abs(gt - item) <= p:
                        batch_sum += 1
                acc_sum += batch_sum / len(y_pred[idx])
            else:
                pred_val = y_pred[idx]
                if abs(gt - pred_val) <= p:
                    acc_sum += 1

        else:
            raise ValueError("Tolerance p not supported!")

        mse_sum += abs(gt - pred_val) ** 2
        mde_sum += abs(gt - pred_val)

    MSE = mse_sum / len(y_gt)
    MDE = mde_sum / len(y_gt)
    Accuracy = acc_sum / len(y_gt)
    print("MSE = %.3f; MDE = %.3f; Accuracy = %.2f%%" % (MSE, MDE, Accuracy * 100))
    return MSE, MDE, Accuracy


if __name__ == '__main__':
    # Load annotations
    with open(join('my_annotations.json')) as f:
        anns = json.load(f)

    # sort anns by Polyphony metric of interest
    sorted_result = eval_polyphony(anns, sort_attr='PolyphonyLevel')
    # X: ; Y_gt: GT count; Y_pred: predicted count
    X, Y_gt, Y_pred = sorted_result[0], sorted_result[1], sorted_result[2]

    mse, mde, accu = get_metric(Y_gt, Y_pred)
