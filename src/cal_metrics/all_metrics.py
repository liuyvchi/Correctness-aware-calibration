import numpy as np
from scipy.optimize import minimize 
from sklearn.metrics import log_loss
import pandas as pd
import time, pdb
from sklearn.metrics import log_loss, brier_score_loss
import sklearn.metrics as metrics
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import average_precision_score, roc_auc_score, auc
import sys
from os import path
# from KDEpy import FFTKDE

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

class _ECELoss():
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.bin_numbers = {}
        

    def eval(self, confidences, accuracies, assign_index=False, cal=False, plot=False):
        confidence_list, accuracy_list, prop_list = [], [], []
        
        ece = np.zeros(1)
        sort_index = np.argsort(confidences)
        # if assign_value:
        #     confidences = confidences[sort_index]
        #     accuracies = accuracies[sort_index]
        bin_number_count = 0
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.__gt__(bin_lower) * confidences.__le__(bin_upper)
            if assign_index:
                self.bin_numbers[str(bin_lower)] = len(confidences[in_bin])
            prop_in_bin = len(confidences[in_bin])/len(confidences)
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].astype(float).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                # if assign_value:
                #     print(accuracy_in_bin)
                #     print(avg_confidence_in_bin)
                bin_ece_pre = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                bin_ece =  bin_ece_pre * prop_in_bin
                ece += bin_ece
                confidence_list.append(avg_confidence_in_bin)
                accuracy_list.append(accuracy_in_bin)
                prop_list.append(prop_in_bin)
            else:
                confidence_list.append(0)
                accuracy_list.append(0)
                prop_list.append(0)
        if plot:
            plot_weighted_histogram(accuracy_list, confidence_list, cal)

        return float(ece.item())

def ensure_numpy(a):
    if not isinstance(a, np.ndarray):
        a = a.numpy()
    return a

def KS_error_from_conf_acc(
    confidences,
    accuracies,
):
    scores = confidences
    labels = accuracies

    scores = ensure_numpy(scores)
    labels = ensure_numpy(labels)

    # Sort the data
    order = scores.argsort()
    scores = scores[order]
    labels = labels[order]

    # Accumulate and normalize by dividing by num samples
    nsamples = len(scores)
    integrated_scores = np.cumsum(scores) / nsamples
    integrated_accuracy = np.cumsum(labels) / nsamples
    # percentile = np.linspace (0.0, 1.0, nsamples)
    # fitted_accuracy, fitted_error = compute_accuracy (scores, labels, spline_method, splines, outdir, plotname, showplots=showplots)

    # Work out the Kolmogorov-Smirnov error
    KS_error_max = np.amax(np.absolute(integrated_scores - integrated_accuracy))

    return KS_error_max

def AdaptiveECE(conf, correctness, conf_bin_num=10):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        ace: expected calibration error
    """
    df = pd.DataFrame({'conf':conf, 'correct':correctness})
    df['correct'] = df['correct'].astype('int')
    df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='quantile').fit_transform(conf[:, np.newaxis])
    
    # groupy by knn + conf
    group_acc = df.groupby(['conf_bin'])['correct'].mean()
    group_confs = df.groupby(['conf_bin'])['conf'].mean()
    counts = df.groupby(['conf_bin'])['conf'].count()
    ace = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
        
    return ace

def MCE(conf, correctness, conf_bin_num = 10):

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        mce: maximum calibration error
    """
    df = pd.DataFrame({'conf':conf, 'correct':correctness})
    df['correct'] = df['correct'].astype('int')

    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))
    # df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='uniform').fit_transform(conf[:, np.newaxis])
    
    # groupy by knn + conf
    group_acc = df.groupby(['conf_bin'])['correct'].mean()
    group_confs = df.groupby(['conf_bin'])['conf'].mean()
    counts = df.groupby(['conf_bin'])['conf'].count()
    mce = (np.abs(group_acc - group_confs) * counts / len(df)).max()
        
    return mce

def brier_score(y_pred, y):
    # y_one_hot = label_binarize(y, classes=np.arange(len(y_pred[0])))
    loss = mean_squared_error(y, y_pred)
    return loss

def all_meausres(confs, correctness, conf_bins = 25, knn_bins=15):
    ece_criterion = _ECELoss(n_bins=25)
    ece = ece_criterion.eval(confs, correctness, assign_index=True)
    bs = brier_score(confs, correctness)
    ace = AdaptiveECE(confs, correctness, conf_bin_num = conf_bins)
    mce = MCE(confs, correctness, conf_bin_num = conf_bins)
    auc = roc_auc_score(correctness, confs)
    ks = KS_error_from_conf_acc(correctness, confs)
    
    return {'ece':ece, 'bs':bs, 'ace':ace, 'mce':mce, 'auc':auc, 'ks':ks}
    