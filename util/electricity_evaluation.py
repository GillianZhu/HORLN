import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import confusion_matrix
import math

def evaluation(result_dataframe, best_threshold=0, trad=False):
    pred = result_dataframe['pred'].values
    flag = result_dataframe['flag'].values.astype(np.int)
    assert (len(pred) == len(flag))
    best_f1_score = 0

    if trad:
        best_f1_threshold = best_threshold
        preds=np.where(pred>best_threshold, 1, 0)
        # best_f1_precision = sum(pred == flag)*1.0 / len(pred)
        conf_matrix = confusion_matrix(flag, preds)
        try:
            tp = conf_matrix[1][1]
        except:
            tp = 0
        try:
            tn = conf_matrix[0][0]
        except:
            t = 0
        try:
            fp = conf_matrix[0][1]
        except:
            fp = 0
        try:
            fn = conf_matrix[1][0]
        except:
            fn = 0

        if (tp != 0 or fp != 0):
            best_f1_precision = tp / (tp + fp)
        else:
            best_f1_precision = 0.0

        if (tp != 0 or fn != 0):
            best_f1_recall = tp / (tp + fn)
        else:
            best_f1_recall = 0.0

        if (best_f1_precision != 0.0 or best_f1_recall != 0):
            best_f1_score = 2 * best_f1_precision * best_f1_recall / (best_f1_precision + best_f1_recall)
    else:
        precisions, recalls, thresholds = precision_recall_curve(flag, pred)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
        best_f1_score_index = np.where(f1_scores == best_f1_score)[0]
        #best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
        best_f1_threshold   = thresholds[best_f1_score_index]
        best_f1_precision   = precisions[best_f1_score_index]
        best_f1_recall      = recalls[best_f1_score_index]

    fpr, tpr, th = roc_curve(flag, pred, pos_label=1)
    elec_auc = auc(fpr, tpr)
    MAP = mean_average_precision(pred, flag, len(flag))

    return best_f1_score, best_f1_threshold, best_f1_precision, best_f1_recall, elec_auc, MAP


def mean_average_precision(pred, flag, topk):
    # pred has been sorted. 
    AP = 0.0
    pos_num = 0.0

    #K = flag.sum()
    #print('K: ', K)
    if topk > len(pred):
        print("topk > all")
        return AP

    for i in range(topk):
        if flag[i] == 1:
            pos_num = pos_num + 1.0
            AP = AP + pos_num / (i + 1)

    assert(pos_num > 0.0), 'pos_num is 0'
    AP = AP / pos_num

    return AP

