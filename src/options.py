import argparse
import torch
import os
from datetime import datetime
import time
import torch 
import random
import numpy as np 
import sys



class Options(object):
    """docstring for Options"""
    def __init__(self):
        super(Options, self).__init__()
        
    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--train_path', type=str, default='/data2/liuyc/ImageNet-val/val', help='training_dataset_path')
        parser.add_argument('--train_prediction_path1', type=str, default='../modelOutput/imagenet_val_out_rotation', help='prediction for tranformation 1')
        parser.add_argument('--train_prediction_path2', type=str, default='../modelOutput/imagenet_val_out_grey', help='prediction for tranformation 2')
        parser.add_argument('--train_prediction_path3', type=str, default='../modelOutput/imagenet_val_out_colorjitter', help='prediction for tranformation 3')

        parser.add_argument('--test_path', type=str, default='/data2/liuyc/imagenet-a', help='test_dataset_path')
        parser.add_argument('--test_prediction_path1', type=str, default='../modelOutput/imagenet_a_out_rotation', help='prediction for tranformation 1')
        parser.add_argument('--test_prediction_path2', type=str, default='../modelOutput/imagenet_a_out_grey', help='prediction for tranformation 2')
        parser.add_argument('--test_prediction_path3', type=str, default='../modelOutput/imagenet_a_out_colorjitter', help='prediction for tranformation 3')

    
        parser.add_argument("--model_name", type=str, default='tv_resnet152', help="model name in timm")
        parser.add_argument('--method_name', type=str, default='3NT4_CA', help='name of method')
        parser.add_argument('--d_name', type=str, default='a', help='name of test set')
        parser.add_argument('--save_dir', type=str, default='./output', help='set out dir')
        

        return parser

    def parse(self):
        parser = self.initialize()
        parser.set_defaults(name=datetime.now().strftime("%y%m%d_%H%M%S"))
        opt = parser.parse_args()


        opt.train_prediction_path1 = opt.train_prediction_path1 + '/%s.npy' % (opt.model_name)
        opt.train_prediction_path2 = opt.train_prediction_path2 + '/%s.npy' % (opt.model_name)
        opt.train_prediction_path3 = opt.train_prediction_path3 + '/%s.npy' % (opt.model_name)
        
        opt.test_prediction_path1 = opt.test_prediction_path1 + '/%s.npy' % (opt.model_name)
        opt.test_prediction_path2 = opt.test_prediction_path2 + '/%s.npy' % (opt.model_name)
        opt.test_prediction_path3 = opt.test_prediction_path3 + '/%s.npy' % (opt.model_name)


        return opt

