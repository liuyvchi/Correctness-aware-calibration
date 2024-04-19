# -*- coding: utf-8 -*-
import os
import cv2
import csv
import math
import random
import numpy as np
import pandas as pd
import argparse
import pickle
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F
import timm

from dataset import ImageNetPredictions, ImageNetPredictions_gt, ImageNetPredictions_gt_half
# from model import Model

from options import Options
import threading
import time

## calibration
from cal_metrics.ECE import _ECELoss
from cal_metrics.BS import brier_score
from cal_metrics.all_metrics import all_meausres
from cal_metrics.temperature_scaling_CA import Temperature, Temperature_adaptive


def print_time(threadName, delay, counter):
    while counter:
        if exitFlag:
            (threading.Thread).exit()
        time.sleep(delay)
        print
        "%s: %s" % (threadName, time.ctime(time.time()))
        counter -= 1


def test(test_loader, valid_loader, device, save_path, d_name='a', m_name='Resnet152', method='3NT4_CA'):
    predicted = []
    gt = []
    ece_criterion = _ECELoss(n_bins=25)

    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    data_num = 0

    d11, d12, d21, d22 = 0, 0, 0, 0
    sd11, sd12, sd21, sd22 = 0, 0, 0, 0
    AC = 0
    AC_vrp_sum = 0

    vicinity1_tmp = []
    vicinity2_tmp = []
    vicinity_predictions1_tmp = []
    vicinity_predictions2_tmp = []
    vicinity_confidence1_tmp = []
    vicinity_confidence2_tmp = []
    index_tmp = []

    seed = 10
    torch.manual_seed(seed)

    logits_valid = []
    Inv_pool = []
    conf_pool = []

    for batch_i, (prediction1_1, prediction1_2, prediction2_1, prediction2_2, prediction3_2, acc, gt, idx) in enumerate(
            valid_loader):
        output1_1 = prediction1_1.to(device)
        output1_2 = prediction1_2.to(device)
        gt = gt.to(device)
        softmax_out1 = F.softmax(output1_1, dim=-1)
        softmax_out2 = F.softmax(output1_2, dim=-1)
        idx = idx.to(device)

        _, predicts1 = torch.topk(output1_1, k=2, dim=1)
        _, predicts2 = torch.topk(output1_2, k=2, dim=1)
        first_predicts1 = predicts1[:, 0]
        first_predicts2 = predicts2[:, 0]
        range_index = torch.tensor(range(len(softmax_out1)))
        confidence_1 = softmax_out1[range_index, first_predicts1]
        confidence_2 = softmax_out2[range_index, first_predicts2]
        correctness = torch.eq(first_predicts1, gt)

        logits_valid.append(output1_1)

        vicinity1_tmp.append(softmax_out1)
        vicinity2_tmp.append(softmax_out2)
        vicinity_predictions1_tmp.append(first_predicts1)
        vicinity_predictions2_tmp.append(first_predicts2)
        vicinity_confidence1_tmp.append(confidence_1)
        vicinity_confidence2_tmp.append(confidence_2)
        index_tmp.append(idx)
        Inv_pool.append(torch.mul(softmax_out1, softmax_out2).sum(dim=-1))
        conf_pool.append(confidence_1)

    index_pool = torch.cat(index_tmp, dim=0)
    length = len(index_pool)

    vicinity1_pool = torch.cat(vicinity1_tmp, dim=0)
    vicinity2_pool = torch.cat(vicinity2_tmp, dim=0)
    vicinity_predictions1_pool = torch.cat(vicinity_predictions1_tmp, dim=0)
    vicinity_predictions2_pool = torch.cat(vicinity_predictions2_tmp, dim=0)
    vicinity_confidence1_pool = torch.cat(vicinity_confidence1_tmp, dim=0)
    vicinity_confidence2_pool = torch.cat(vicinity_confidence2_tmp, dim=0)
    logits_valid_pool = torch.cat(logits_valid, dim=0)
    Inv_pool = torch.cat(Inv_pool, dim=0)
    conf_pool = torch.cat(conf_pool, dim=0)

    logits_v_valid = []
    logits_valid = []
    logits_valid_2 = []
    logits_valid_3 = []
    logits_valid_4 = []
    valid_labels = []
    Inv_valid = []
    conf1_valid = []
    conf2_valid = []
    conf1_t2_valid = []
    conf2_t2_valid = []
    conf12_valid = []
    conf21_valid = []
    conf_v_valid = []
    
    p2topk1_valid = []
    p3topk1_valid = []
    p4topk1_valid = []
    
    p1topk2_valid = []
    p3topk2_valid = []
    p4topk2_valid = []
    
    p1topk3_valid = []
    p2topk3_valid = []
    p4topk3_valid = []
    
    p1topk4_valid = []
    p2topk4_valid = []
    p3topk4_valid = []
    
    pN_on1_valid = []
    pN_on2_valid = []
    pN_on3_valid = []
    pN_on4_valid = []
    var2on1_valid = []
    var3on1_valid = []
    var4on1_valid = []

    Inv_v_valid = []
    for batch_i, (prediction1_1, prediction1_2, prediction2_1, prediction2_2, prediction3_2, acc, gt, idx) in enumerate(
            valid_loader):
        valid_labels.append(gt.cuda())
        output1_1 = prediction1_1.to(device)
        logits_valid.append(output1_1)
        _, top_idx_1 = torch.topk(output1_1, k=4, dim=-1)
        output1_2 = prediction1_2.to(device)
        logits_valid_2.append(output1_2)
        _, top_idx_2 = torch.topk(output1_2, k=4, dim=-1)
        output1_3 = prediction2_2.to(device)
        logits_valid_3.append(output1_3)
        _, top_idx_3 = torch.topk(output1_3, k=4, dim=-1)
        output1_4 = prediction3_2.to(device)
        logits_valid_4.append(output1_4)
        _, top_idx_4 = torch.topk(output1_4, k=4, dim=-1)

        softmax_out1 = F.softmax(output1_1, dim=-1)
        softmax_out2 = F.softmax(output1_2, dim=-1)
        softmax_out3 = F.softmax(output1_3, dim=-1)
        softmax_out4 = F.softmax(output1_4, dim=-1)
        confidence1, confidence1_t2 = softmax_out1.max(dim=-1)[0], softmax_out1.max(dim=-1)[1]
        confidence2, confidence2_t2 = softmax_out2.max(dim=-1)[0], softmax_out1.max(dim=-1)[1]
        idx = idx.to(device)

        _, predicts1 = torch.topk(output1_1, k=2, dim=1)
        _, predicts2 = torch.topk(output1_2, k=2, dim=1)
        _, predicts3 = torch.topk(output1_3, k=2, dim=1)
        first_predicts1 = predicts1[:, 0]
        first_predicts2 = predicts2[:, 0]
        same_pred = torch.eq(first_predicts1, first_predicts2)
        notsame_pred = ~same_pred

        index = index_pool

        range_index = torch.tensor(range(len(softmax_out1)))
        p12 = softmax_out1[range_index, first_predicts2]
        p21 = softmax_out2[range_index, first_predicts1]

        p1topk2 = torch.gather(softmax_out1, 1, top_idx_2)
        p1topk3 = torch.gather(softmax_out1, 1, top_idx_3)
        p1topk4 = torch.gather(softmax_out1, 1, top_idx_4)
        
        p2topk1 = torch.gather(softmax_out2, 1, top_idx_1)
        p2topk3 = torch.gather(softmax_out2, 1, top_idx_3)
        p2topk4 = torch.gather(softmax_out2, 1, top_idx_4)
        
        p3topk1 = torch.gather(softmax_out3, 1, top_idx_1)
        p3topk2 = torch.gather(softmax_out3, 1, top_idx_2)
        p3topk4 = torch.gather(softmax_out3, 1, top_idx_4)
        
        p4topk1 = torch.gather(softmax_out4, 1, top_idx_1)
        p4topk2 = torch.gather(softmax_out4, 1, top_idx_2)
        p4topk3 = torch.gather(softmax_out4, 1, top_idx_3)

        var2on1 = -2 * p1topk2.mul(p2topk1) + p1topk2 + p2topk1
        var3on1 = -2 * p1topk3.mul(p3topk1) + p1topk3 + p3topk1
        var4on1 = -2 * p1topk4.mul(p4topk1) + p1topk4 + p4topk1

        share_idx = torch.tensor(range(len(output1_1)))
        pN_on1 = torch.stack(
            (softmax_out2[share_idx, top_idx_1[:, 0]],
             softmax_out3[share_idx, top_idx_1[:, 0]], softmax_out4[share_idx, top_idx_1[:, 0]]),
            dim=-1)
        pN_on2 = torch.stack(
            (softmax_out1[share_idx, top_idx_2[:, 0]],
             softmax_out3[share_idx, top_idx_2[:, 0]], softmax_out4[share_idx, top_idx_2[:, 0]]),
            dim=-1)
        pN_on3 = torch.stack(
            (softmax_out1[share_idx, top_idx_3[:, 0]], softmax_out2[share_idx, top_idx_3[:, 0]],
             softmax_out4[share_idx, top_idx_3[:, 0]]),
            dim=-1)
        pN_on4 = torch.stack(
            (softmax_out1[share_idx, top_idx_4[:, 0]], softmax_out2[share_idx, top_idx_4[:, 0]],
             softmax_out3[share_idx, top_idx_4[:, 0]]),
            dim=-1)

        mask_self = (idx.unsqueeze(1) == index.unsqueeze(0)).long()


        Inv_valid.append(torch.mul(softmax_out1, softmax_out2).sum(dim=-1))
        conf1_valid.append(confidence1)
        conf2_valid.append(confidence2)
        conf1_t2_valid.append(confidence1_t2)
        conf2_t2_valid.append(confidence2_t2)
        conf12_valid.append(p12)
        conf21_valid.append(p21)
       
        p2topk1_valid.append(p2topk1)
        p3topk1_valid.append(p3topk1)
        p4topk1_valid.append(p4topk1)
        
        p1topk2_valid.append(p1topk2)
        p3topk2_valid.append(p3topk2)
        p4topk2_valid.append(p4topk2)
        
        p1topk3_valid.append(p1topk3)
        p2topk3_valid.append(p2topk3)
        p4topk3_valid.append(p4topk3)
        
        p1topk4_valid.append(p1topk4)
        p2topk4_valid.append(p2topk4)
        p3topk4_valid.append(p3topk4)
        
        pN_on1_valid.append(pN_on1)
        pN_on2_valid.append(pN_on2)
        pN_on3_valid.append(pN_on3)
        pN_on4_valid.append(pN_on4)
        var2on1_valid.append(var2on1)
        var3on1_valid.append(var3on1)
        var4on1_valid.append(var4on1)

        iter_cnt += 1
        data_num += len(softmax_out1)


    logits_valid = torch.cat(logits_valid, dim=0)
    logits_valid_2 = torch.cat(logits_valid_2, dim=0)
    valid_labels = torch.cat(valid_labels, dim=0)

    pN_on1_valid = torch.cat(pN_on1_valid, dim=0)
    pN_on2_valid = torch.cat(pN_on2_valid, dim=0)
    pN_on3_valid = torch.cat(pN_on3_valid, dim=0)
    pN_on4_valid = torch.cat(pN_on4_valid, dim=0)

    p2topk1_valid = torch.cat(p2topk1_valid, dim=0)
    p3topk1_valid = torch.cat(p3topk1_valid, dim=0)
    p4topk1_valid = torch.cat(p4topk1_valid, dim=0)
    
    p1topk2_valid = torch.cat(p1topk2_valid, dim=0)
    p3topk2_valid = torch.cat(p3topk2_valid, dim=0)
    p4topk2_valid = torch.cat(p4topk2_valid, dim=0)
    
    p1topk3_valid = torch.cat(p1topk3_valid, dim=0)
    p2topk3_valid = torch.cat(p2topk3_valid, dim=0)
    p4topk3_valid = torch.cat(p4topk3_valid, dim=0)
    
    p1topk4_valid = torch.cat(p1topk4_valid, dim=0)
    p2topk4_valid = torch.cat(p2topk4_valid, dim=0)
    p3topk4_valid = torch.cat(p3topk4_valid, dim=0)

    var2on1_valid = torch.cat(var2on1_valid, dim=0)
    var3on1_valid = torch.cat(var3on1_valid, dim=0)
    var4on1_valid = torch.cat(var4on1_valid, dim=0)

    top_k_cues, _ = torch.topk(logits_valid, k=10, dim=-1)
    top_k_cues_2, _ = torch.topk(logits_valid_2, k=4, dim=-1)

    temp_args = torch.cat([p2topk1_valid, p3topk1_valid, p4topk1_valid], dim=-1)

    # calibration
    temperature_model = Temperature_adaptive(input_dim=12, feat_dim=5)
    TS_model = Temperature()
    load_model = False
    if load_model:
        # Load saved model weights
        temperature_model.load_state_dict(torch.load('ckpt/model_weights_%s_%s.pth' % (m_name, method)))
        temperature_model.cuda()
        TS_model.cuda()
        TS_model.temperature = temperature_model.temperature
    else:
        temperature_model.train()
        temperature_model.set_temperature(logits_valid.detach().to(device), temp_args, valid_labels.detach().to(device))

        temperature_model.temperature = TS_model.temperature
        # Save model weights
        torch.save(temperature_model.state_dict(), 'ckpt/model_weights_%s_%s.pth' % (m_name, method))
    del logits_v_valid
    del valid_labels
    temperature_model.eval()
    TS_model.eval()

    ################

    # test ece
    logits_test_pool = []
    vicinity1_tmp = []
    vicinity2_tmp = []
    vicinity_predictions1_tmp = []
    vicinity_predictions2_tmp = []
    vicinity_confidence1_tmp = []
    vicinity_confidence2_tmp = []
    index_tmp = []
    Inv_pool = []
    conf_pool = []

    seed = 10
    torch.manual_seed(seed)

    for batch_i, (prediction1_1, prediction1_2, prediction2_1, prediction2_2, prediction3_2, acc, gt, idx) in enumerate(
            test_loader):
        output1_1 = prediction1_1.to(device)
        output1_2 = prediction1_2.to(device)
        gt = gt.to(device)
        softmax_out1 = F.softmax(output1_1, dim=-1)
        softmax_out2 = F.softmax(output1_2, dim=-1)
        idx = idx.to(device)

        _, predicts1 = torch.topk(output1_1, k=2, dim=1)
        _, predicts2 = torch.topk(output1_2, k=2, dim=1)
        first_predicts1 = predicts1[:, 0]
        first_predicts2 = predicts2[:, 0]
        range_index = torch.tensor(range(len(softmax_out1)))
        confidence_1 = softmax_out1[range_index, first_predicts1]
        confidence_2 = softmax_out2[range_index, first_predicts2]

        logits_test_pool.append(output1_1)
        vicinity1_tmp.append(softmax_out1)
        vicinity2_tmp.append(softmax_out2)
        vicinity_predictions1_tmp.append(first_predicts1)
        vicinity_predictions2_tmp.append(first_predicts2)
        vicinity_confidence1_tmp.append(confidence_1)
        vicinity_confidence2_tmp.append(confidence_2)

        index_tmp.append(idx)
        Inv_pool.append(torch.mul(softmax_out1, softmax_out2).sum(dim=-1))
        conf_pool.append(confidence_1)

    index_pool = torch.cat(index_tmp, dim=0)
    length = len(index_pool)

    vicinity1_pool = torch.cat(vicinity1_tmp, dim=0)
    vicinity2_pool = torch.cat(vicinity2_tmp, dim=0)
    vicinity_predictions1_pool = torch.cat(vicinity_predictions1_tmp, dim=0)
    vicinity_predictions2_pool = torch.cat(vicinity_predictions2_tmp, dim=0)
    vicinity_confidence1_pool = torch.cat(vicinity_confidence1_tmp, dim=0)
    vicinity_confidence2_pool = torch.cat(vicinity_confidence2_tmp, dim=0)
    logits_test_pool = torch.cat(logits_test_pool, dim=0)
    Inv_pool = torch.cat(Inv_pool, dim=0)
    conf_pool = torch.cat(conf_pool, dim=0)

    correctness_test = []
    logits_v_test = []
    logits_test = []
    logits_test_2 = []
    logits_test_3 = []
    logits_test_4 = []
    test_labels = []
    Inv_test = []
    conf1_test = []
    conf2_test = []
    conf1_t2_test = []
    conf2_t2_test = []
    conf12_test = []
    conf21_test = []

    Inv_v_test = []
    conf_v_test = []
    
    p2topk1_test = []
    p3topk1_test = []
    p4topk1_test = []
    
    p1topk2_test = []
    p3topk2_test = []
    p4topk2_test = []
    
    p1topk3_test = []
    p2topk3_test = []
    p4topk3_test = []
    
    p1topk4_test = []
    p2topk4_test = []
    p3topk4_test = []
    
    pN_on1_test = []
    pN_on2_test = []
    pN_on3_test = []
    pN_on4_test = []

    var2on1_test = []
    var3on1_test = []
    var4on1_test = []

    for batch_i, (prediction1_1, prediction1_2, prediction2_1, prediction2_2, prediction3_2, acc, gt, idx) in enumerate(
            test_loader):
        gt = gt.to(device)
        test_labels.append(gt.cpu())
        output1_1 = prediction1_1.to(device)
        logits_test.append(output1_1)
        _, top_idx_1 = torch.topk(output1_1, k=4, dim=-1)
        output1_2 = prediction1_2.to(device)
        logits_test_2.append(output1_2)
        _, top_idx_2 = torch.topk(output1_2, k=4, dim=-1)
        output1_3 = prediction2_2.to(device)
        logits_test_3.append(output1_3)
        _, top_idx_3 = torch.topk(output1_3, k=4, dim=-1)
        output1_4 = prediction3_2.to(device)
        logits_test_4.append(output1_4)
        _, top_idx_4 = torch.topk(output1_4, k=4, dim=-1)

        softmax_out1 = F.softmax(output1_1, dim=-1)
        softmax_out2 = F.softmax(output1_2, dim=-1)
        softmax_out3 = F.softmax(output1_3, dim=-1)
        softmax_out4 = F.softmax(output1_4, dim=-1)
        confidence1, confidence1_t2 = softmax_out1.max(dim=-1)[0], softmax_out1.max(dim=-1)[1]
        confidence2, confidence2_t2 = softmax_out2.max(dim=-1)[0], softmax_out1.max(dim=-1)[1]
        idx = idx.to(device)

        _, predicts1 = torch.topk(output1_1, k=2, dim=1)
        _, predicts2 = torch.topk(output1_2, k=2, dim=1)
        first_predicts1 = predicts1[:, 0]
        first_predicts2 = predicts2[:, 0]
        same_pred = torch.eq(first_predicts1, first_predicts2)
        notsame_pred = ~same_pred
        correctness = torch.eq(first_predicts1, gt)
        correctness_test.append(correctness.cpu())

        index = index_pool
        vicinity1 = vicinity1_pool
        vicinity2 = vicinity2_pool
        vicinity_predictions1 = vicinity_predictions1_pool
        vicinity_predictions2 = vicinity_predictions2_pool
        vicinity_confidence1 = vicinity_confidence1_pool
        vicinity_confidence2 = vicinity_confidence2_pool

        range_index = torch.tensor(range(len(softmax_out1)))
        p12 = softmax_out1[range_index, first_predicts2]
        p21 = softmax_out2[range_index, first_predicts1]

        p1topk2 = torch.gather(softmax_out1, 1, top_idx_2)
        p1topk3 = torch.gather(softmax_out1, 1, top_idx_3)
        p1topk4 = torch.gather(softmax_out1, 1, top_idx_4)
        
        p2topk1 = torch.gather(softmax_out2, 1, top_idx_1)
        p2topk3 = torch.gather(softmax_out2, 1, top_idx_3)
        p2topk4 = torch.gather(softmax_out2, 1, top_idx_4)
        
        p3topk1 = torch.gather(softmax_out3, 1, top_idx_1)
        p3topk2 = torch.gather(softmax_out3, 1, top_idx_2)
        p3topk4 = torch.gather(softmax_out3, 1, top_idx_4)
        
        p4topk1 = torch.gather(softmax_out4, 1, top_idx_1)
        p4topk2 = torch.gather(softmax_out4, 1, top_idx_2)
        p4topk3 = torch.gather(softmax_out4, 1, top_idx_3)

        var2on1 = -2*p1topk2.mul(p2topk1) + p1topk2 + p2topk1
        var3on1 = -2*p1topk3.mul(p3topk1) + p1topk3 + p3topk1
        var4on1 = -2*p1topk4.mul(p4topk1) + p1topk4 + p4topk1

        share_idx = torch.tensor(range(len(output1_1)))
        pN_on1 = torch.stack(
            (softmax_out2[share_idx, top_idx_1[:, 0]],
             softmax_out3[share_idx, top_idx_1[:, 0]], softmax_out4[share_idx, top_idx_1[:, 0]]),
            dim=-1)
        pN_on2 = torch.stack(
            (softmax_out1[share_idx, top_idx_2[:, 0]],
             softmax_out3[share_idx, top_idx_2[:, 0]], softmax_out4[share_idx, top_idx_2[:, 0]]),
            dim=-1)
        pN_on3 = torch.stack(
            (softmax_out1[share_idx, top_idx_3[:, 0]], softmax_out2[share_idx, top_idx_3[:, 0]],
             softmax_out4[share_idx, top_idx_3[:, 0]]),
            dim=-1)
        pN_on4 = torch.stack(
            (softmax_out1[share_idx, top_idx_4[:, 0]], softmax_out2[share_idx, top_idx_4[:, 0]],
             softmax_out3[share_idx, top_idx_4[:, 0]]),
            dim=-1)

        mask_self = (idx.unsqueeze(1) == index.unsqueeze(0)).long()

        ## autoEval measruement

        Inv_test.append(torch.mul(softmax_out1, softmax_out2).sum(dim=-1))
        conf1_test.append(confidence1)
        conf2_test.append(confidence2)
        conf1_t2_test.append(confidence1_t2)
        conf2_t2_test.append(confidence2_t2)
        conf12_test.append(p12)
        conf21_test.append(p21)
        
        p2topk1_test.append(p2topk1)
        p3topk1_test.append(p3topk1)
        p4topk1_test.append(p4topk1)
        
        p1topk2_test.append(p1topk2)
        p3topk2_test.append(p3topk2)
        p4topk2_test.append(p4topk2)
        
        p1topk3_test.append(p1topk3)
        p2topk3_test.append(p2topk3)
        p4topk3_test.append(p4topk3)
        
        p1topk4_test.append(p1topk4)
        p2topk4_test.append(p2topk4)
        p3topk4_test.append(p3topk4)
        
        pN_on1_test.append(pN_on1)
        pN_on2_test.append(pN_on2)
        pN_on3_test.append(pN_on3)
        pN_on4_test.append(pN_on4)
        var2on1_test.append(var2on1)
        var3on1_test.append(var3on1)
        var4on1_test.append(var4on1)

        iter_cnt += 1
        data_num += len(softmax_out1)

    logits_test = torch.cat(logits_test, dim=0)
    correctness_test = torch.cat(correctness_test, dim=0)

    pN_on1_test = torch.cat(pN_on1_test, dim=0)
    pN_on2_test = torch.cat(pN_on2_test, dim=0)
    pN_on3_test = torch.cat(pN_on3_test, dim=0)
    pN_on4_test = torch.cat(pN_on4_test, dim=0)

    p2topk1_test = torch.cat(p2topk1_test, dim=0)
    p3topk1_test = torch.cat(p3topk1_test, dim=0)
    p4topk1_test = torch.cat(p4topk1_test, dim=0)
    
    p1topk2_test = torch.cat(p1topk2_test, dim=0)
    p3topk2_test = torch.cat(p3topk2_test, dim=0)
    p4topk2_test = torch.cat(p4topk2_test, dim=0)
    
    p1topk3_test = torch.cat(p1topk3_test, dim=0)
    p2topk3_test = torch.cat(p2topk3_test, dim=0)
    p4topk3_test = torch.cat(p4topk3_test, dim=0)
    
    p1topk4_test = torch.cat(p1topk4_test, dim=0)
    p2topk4_test = torch.cat(p2topk4_test, dim=0)
    p3topk4_test = torch.cat(p3topk4_test, dim=0)

    var2on1_test = torch.cat(var2on1_test, dim=0)
    var3on1_test = torch.cat(var3on1_test, dim=0)
    var4on1_test = torch.cat(var4on1_test, dim=0)


    top_k_cues, _ = torch.topk(logits_test, k=10, dim=-1)
    test_labels=torch.cat(test_labels, dim=0)
    
    conf_test = F.softmax(logits_test, dim=-1).max(dim=-1)[0]
    conf_gt = F.softmax(logits_test, dim=-1)[torch.arange(len(test_labels)), test_labels]

    temp_args = torch.cat([p2topk1_test, p3topk1_test, p4topk1_test], dim=-1)

    

    print("uncal")
    softmax_test = F.softmax(logits_test, dim=-1).cpu()
    confidence_test = softmax_test.max(dim=-1)[0]
    
    #compute measurement for before calibration
    erros_uncal = all_meausres(confidence_test.numpy(), correctness_test.numpy())
    temperature_model.eval()

    # logits_cal_test = TS_model(logits_test.cuda())
    logits_cal_test, temp, correctness_out = temperature_model(logits_test.cuda(), temp_args, return_temp=True, correctness=True)
    
    softmax_cal_test = F.softmax(logits_cal_test, dim=-1).detach().cpu()
    confidence_cal_test = softmax_cal_test.max(dim=-1)[0]
    correctness_acc = ((correctness_out.cpu()>0.5).long() == correctness_test.long()).sum()/len(correctness_test)
    

    print("cal")
    #compute measurement for after calibration
    erros_cal = all_meausres(confidence_cal_test.numpy(), correctness_test.numpy())

    # save data to pickle file
    with open('%s/%s_%s_%s_temperature.pkl' % (save_path, d_name, m_name, method), 'wb') as f:
        pickle.dump(temp.detach().cpu().numpy(), f)

    with open('%s/%s_%s_correctness_test.pkl' % (save_path, d_name, m_name), 'wb') as f1:
        pickle.dump(correctness_test.cpu().numpy(), f1)

    with open('%s/%s_%s_confgt.pkl' % (save_path, d_name, m_name), 'wb') as f1:
        pickle.dump(conf_gt.cpu().numpy(), f1)
    
    with open('%s/%s_%s_%s_conf.pkl' % (save_path, d_name, m_name, method), 'wb') as f1, open(
            '%s/%s_%s_uncal_conf.pkl' % (save_path, d_name, m_name), 'wb') as f2:
        pickle.dump(confidence_cal_test.cpu().numpy(), f1)
        pickle.dump(confidence_test.cpu().numpy(), f2)


    return erros_uncal, erros_cal, correctness_acc


def main(args):
    # setup_seed(0)
    
    model_name = args.model_name
    d_name = args.d_name
    method_name = args.method_name

    save_path = args.save_dir

    print(model_name)

    

    test_dataset = ImageNetPredictions_gt(args.test_path, args.test_prediction_path1, args.test_prediction_path2, args.test_prediction_path3)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 drop_last=False)

    validation_set = ImageNetPredictions_gt(args.train_path, args.train_prediction_path1, args.train_prediction_path2, args.train_prediction_path3)
    validation_loader = torch.utils.data.DataLoader(validation_set,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    drop_last=False)


    start_time = time.time()
    device = torch.device('cuda:0')
    erros_uncal, erros_cal, correctness_acc = test(test_loader, validation_loader, device, save_path=save_path, d_name=d_name, m_name=model_name, method=method_name)
    print("--- %s seconds ---" % (time.time() - start_time))

    print(erros_uncal, erros_cal, correctness_acc)

    if os.path.exists('results/%s_%s.pkl' % (method_name, d_name)):

        with open('results/%s_%s.pkl' % (method_name, d_name), 'rb') as file:
            my_dict = pickle.load(file)
    else:
        my_dict = {}

    my_dict[model_name] = [erros_uncal, erros_cal]
    with open('results/%s_%s.pkl' % (method_name, d_name), 'wb') as file:
        pickle.dump(my_dict, file)
    
opt = Options().parse()
main(opt)
