# -*- coding: utf-8 -*-
import os
import cv2
import torch
import torch.utils.data as data
import pandas as pd
import random
from torchvision import transforms, datasets

import pandas as pd
from PIL import Image
import glob
import numpy as np

import torch.nn.functional as F

def AUs_dropout(ganimation_model, imgs):
    pred_AUs = ganimation_model(imgs)
    erased_Aus = F.dropout(pred_AUs, p=0.5)
    erased_imgs = ganimation_model(erased_Aus, imgs)
    return erased_imgs

def get_ImageNetV2(dataset_folder, transform):
    dataset = datasets.ImageFolder(root=dataset_folder, transform=transform)
    return dataset

class ImageNetPredictions_gt(data.Dataset):
    def __init__(self, ImageNetV2_path, prediction_path1, prediction_path2, prediction_path3, type='grey'):
        self.ImageNetV2_path = ImageNetV2_path
        self.prediction_path1 = prediction_path1
        self.prediction_path2 = prediction_path2
        self.prediction_path3 = prediction_path3
        self.type = type
        self.prediction_dic1 = np.load(self.prediction_path1, allow_pickle=True).item()
        self.prediction_dic2 = np.load(self.prediction_path2, allow_pickle=True).item()
        self.prediction_dic3 = np.load(self.prediction_path3, allow_pickle=True).item()
        self.acc = self.prediction_dic1['acc'].item()
        self.keys = self.prediction_dic1.keys()
        self.leng = len(self.keys)-1
        
        self.label = []
        dirs = os.listdir(self.ImageNetV2_path)
        dirs.sort()
        label_flag = 0
        for dir in dirs:
            dir_path = os.path.join(self.ImageNetV2_path, dir)
            if not os.path.isdir(dir_path):
                continue
            img_names = os.listdir(dir_path)
            for img in img_names:
                if 'imagenetv2' in self.ImageNetV2_path:
                    self.label.append(int(dir))
                elif 'objectnet' in self.ImageNetV2_path:
                    if label_flag in ObjectNet_to_113.keys():
                        self.label.append(ObjectNet_to_113[label_flag])
                    else: 
                        break
                else:
                    self.label.append(label_flag)
            label_flag+=1

    def __len__(self):
        return len(self.keys)-1

    def __getitem__(self, idx):
        list1 = self.prediction_dic1[str(torch.tensor(idx))]
        list2 = self.prediction_dic2[str(torch.tensor(idx))]
        list3 = self.prediction_dic3[str(torch.tensor(idx))]
        try:
            gt = self.label[idx]
        except:
            print(len(self.label))
            print(idx)
            assert(0)
    
        if self.type == 'grey':
            prediction1_1 = list1[0]
            prediction1_2 = list1[1]
            prediction2_1 = list2[0]
            prediction2_2 = list2[1]
            prediction3_2 = list3[1]
            return prediction1_1, prediction1_2, prediction2_1, prediction2_2, prediction3_2, self.acc, gt, idx
        elif self.type == 'mixup':
            prediction1_1 = list1[0]
            prediction1_2 = list1[1]
            lamd = list1[2]
            mix_idx = list1[3].item()
            prediction2_1 = list2[0]
            prediction2_2 = list2[1]
            return prediction1_1, prediction1_2, prediction2_1, prediction2_2, lamd, mix_idx, self.acc, gt, idx
        

class ImageNetPredictions_gt_half(data.Dataset):
    def __init__(self, ImageNetV2_path, prediction_path1, prediction_path2, prediction_path3, type='grey', half='train'):
        self.ImageNetV2_path = ImageNetV2_path
        self.prediction_path1 = prediction_path1
        self.prediction_path2 = prediction_path2
        self.prediction_path3 = prediction_path3
        self.type = type
        self.prediction_dic1 = np.load(self.prediction_path1, allow_pickle=True).item()
        self.prediction_dic2 = np.load(self.prediction_path2, allow_pickle=True).item()
        self.prediction_dic3 = np.load(self.prediction_path3, allow_pickle=True).item()
        self.acc = self.prediction_dic1['acc'].item()
        self.keys = self.prediction_dic1.keys()
        self.half=half
        
        
        self.label = []
        dirs = os.listdir(self.ImageNetV2_path)
        dirs.sort()
        label_flag = 0
        for dir in dirs:
            dir_path = os.path.join(self.ImageNetV2_path, dir)
            if not os.path.isdir(dir_path):
                continue
            img_names = os.listdir(dir_path)
            for img in img_names:
                if 'imagenetv2' in self.ImageNetV2_path:
                    self.label.append(int(dir))
                elif 'objectnet' in self.ImageNetV2_path:
                    if label_flag in ObjectNet_to_113.keys():
                        self.label.append(ObjectNet_to_113[label_flag])
                    else: 
                        break
                else:
                    self.label.append(label_flag)
            label_flag+=1
            
        self.leng = int((len(self.keys)-1)/2)-1

    def __len__(self):
        return int((len(self.keys)-1)/2)-1

    def __getitem__(self, idx):
        if self.half=='train':
            idx = idx
        elif self.half=='test':
            idx = self.leng + idx
        list1 = self.prediction_dic1[str(torch.tensor(idx))]
        list2 = self.prediction_dic2[str(torch.tensor(idx))]
        list3 = self.prediction_dic3[str(torch.tensor(idx))]
        try:
            gt = self.label[idx]
        except:
            print(len(self.label))
            print(idx)
            assert(0)
    
        if self.type == 'grey':
            prediction1_1 = list1[0]
            prediction1_2 = list1[1]
            prediction2_1 = list2[0]
            prediction2_2 = list2[1]
            prediction3_2 = list3[1]
            return prediction1_1, prediction1_2, prediction2_1, prediction2_2, prediction3_2, self.acc, gt, idx
        elif self.type == 'mixup':
            prediction1_1 = list1[0]
            prediction1_2 = list1[1]
            lamd = list1[2]
            mix_idx = list1[3].item()
            prediction2_1 = list2[0]
            prediction2_2 = list2[1]
            return prediction1_1, prediction1_2, prediction2_1, prediction2_2, lamd, mix_idx, self.acc, gt, idx



class ImageNetPredictions(data.Dataset):
    def __init__(self, ImageNetV2_path, prediction_path1, prediction_path2, type='grey', set_portion=1):
        self.ImageNetV2_path = ImageNetV2_path
        self.prediction_path1 = prediction_path1
        self.prediction_path2 = prediction_path2
        self.type = type
        self.prediction_dic1 = np.load(self.prediction_path1, allow_pickle=True).item()
        self.prediction_dic2 = np.load(self.prediction_path2, allow_pickle=True).item()
        self.acc = self.prediction_dic1['acc'].item()
        self.keys = self.prediction_dic1.keys()
        self.set_portion = set_portion
        self.label = []
        # dirs = os.listdir(self.ImageNetV2_path)
        # dirs.sort()
        # label_flag = 0
        # for dir in dirs:
        #     dir_path = os.path.join(self.ImageNetV2_path, dir)
        #     if not os.path.isdir(dir_path):
        #         continue
        #     img_names = os.listdir(dir_path)
        #     for img in img_names:
        #         if 'imagenetv2' in self.ImageNetV2_path:
        #             self.label.append(int(dir))
        #         else:
        #             self.label.append(label_flag)
        #     label_flag+=1

    def __len__(self):
        return int((len(self.keys)-1)*(self.set_portion))

    def __getitem__(self, idx):
        # # cifar10 test plugin
        # list1 = self.prediction_dic1[str(idx)]
        # list2 = self.prediction_dic2[str(idx)]

        list1 = self.prediction_dic1[str(torch.tensor(idx))]
        list2 = self.prediction_dic2[str(torch.tensor(idx))]
        # gt = self.label[idx]
        if self.type == 'grey':
            prediction1_1 = list1[0]
            prediction1_2 = list1[1]
            prediction2_1 = list2[0]
            prediction2_2 = list2[1]
            return prediction1_1, prediction1_2, prediction2_1, prediction2_2, self.acc, idx
        elif self.type == 'mixup':
            prediction1_1 = list1[0]
            prediction1_2 = list1[1]
            lamd = list1[2]
            mix_idx = list1[3].item()
            prediction2_1 = list2[0]
            prediction2_2 = list2[1]
            return prediction1_1, prediction1_2, prediction2_1, prediction2_2, lamd, mix_idx, self.acc, idx

class ImageNetV(data.Dataset):
    def __init__(self, ImageNetV2_path, transform=None, without_aug=False):
        self.ImageNetV2_path = ImageNetV2_path
        self.transform1 = transform
        self.file_paths = []
        self.label = []
        dirs = os.listdir(self.ImageNetV2_path)
        dirs.sort()
        label_flag = 0
        for dir in dirs:
            dir_path = os.path.join(self.ImageNetV2_path, dir)
            if not os.path.isdir(dir_path):
                continue
            img_names = os.listdir(dir_path)
            for img in img_names:       
                if 'imagenetv2' in self.ImageNetV2_path:
                    self.label.append(int(dir))
                elif 'objectnet' in self.ImageNetV2_path:
                    if label_flag in ObjectNet_to_113.keys():
                        self.label.append(ObjectNet_to_113[label_flag])
                    else: 
                        break
                else:
                    self.label.append(label_flag)
                img_path = os.path.join(dir_path, img)
                self.file_paths.append(img_path)
            label_flag+=1

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])

        image = image[:, :, ::-1]

        image1 = self.transform1(image)

        return image1, label


class ImageNetV_twoTransforms(data.Dataset):
    def __init__(self, ImageNetV2_path, transform=None, transform2=None, without_aug=False):
        self.ImageNetV2_path = ImageNetV2_path
        self.transform1 = transform
        self.transform2 = transform2
        self.file_paths = []
        self.label = []
        dirs = os.listdir(self.ImageNetV2_path)
        dirs.sort()
        label_flag = 0
        for dir in dirs:
            dir_path = os.path.join(self.ImageNetV2_path, dir)
            if not os.path.isdir(dir_path):
                continue
            img_names = os.listdir(dir_path)
            for img in img_names:       
                if 'imagenetv2' in self.ImageNetV2_path:
                    self.label.append(int(dir))
                elif 'objectnet' in self.ImageNetV2_path:
                    if label_flag in ObjectNet_to_113.keys():
                        self.label.append(ObjectNet_to_113[label_flag])
                    else: 
                        break
                else:
                    self.label.append(label_flag)
                img_path = os.path.join(dir_path, img)
                self.file_paths.append(img_path)
            label_flag+=1

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            label = self.label[idx]
            image = cv2.imread(self.file_paths[idx])

            image = image[:, :, ::-1]

            image2 = self.transform2(image)

            image1 = self.transform1(image)
        except:
            print(self.file_paths[idx])
            assert(0)


        return image1, image2, label, idx


class iwildcamPredictions_gt(data.Dataset):
    def __init__(self, ImageNetV2_path, prediction_path1, prediction_path2, prediction_path3, split, type='grey'):
        self.ImageNetV2_path = ImageNetV2_path
        self.prediction_path1 = prediction_path1
        self.prediction_path2 = prediction_path2
        self.prediction_path3 = prediction_path3
        self.type = type
        self.prediction_dic1 = np.load(self.prediction_path1, allow_pickle=True).item()
        self.prediction_dic2 = np.load(self.prediction_path2, allow_pickle=True).item()
        self.prediction_dic3 = np.load(self.prediction_path3, allow_pickle=True).item()
        self.acc = self.prediction_dic1['acc'].item()
        self.keys = self.prediction_dic1.keys()
        
        self.label = []
        dirs = os.listdir(self.ImageNetV2_path)
        dirs.sort()
        label_flag = 0
        self.metadata_path = '/data2/liuyc/iwildcam_v2.0/metadata.csv'
        self.metadata_pd = pd.read_csv(self.metadata_path) 
        self.test_pd = self.metadata_pd[self.metadata_pd['split']==split]
        self._n_classes = max(self.metadata_pd['y']) + 1
        assert len(np.unique(self.metadata_pd['y'])) == self._n_classes

        for index, row in self.test_pd.iterrows():
            filename = row['filename']
            y = row['y']
            self.label.append(y)


    def __len__(self):
        return len(self.keys)-1

    def __getitem__(self, idx):
        list1 = self.prediction_dic1[str(torch.tensor(idx))]
        list2 = self.prediction_dic2[str(torch.tensor(idx))]
        list3 = self.prediction_dic3[str(torch.tensor(idx))]
        try:
            gt = self.label[idx]
        except:
            print(len(self.label))
            print(idx)
            assert(0)
    
        if self.type == 'grey':
            prediction1_1 = list1[0]
            prediction1_2 = list1[1]
            prediction2_1 = list2[0]
            prediction2_2 = list2[1]
            prediction3_2 = list3[1]
            return prediction1_1, prediction1_2, prediction2_1, prediction2_2, prediction3_2, self.acc, gt, idx
        elif self.type == 'mixup':
            prediction1_1 = list1[0]
            prediction1_2 = list1[1]
            lamd = list1[2]
            mix_idx = list1[3].item()
            prediction2_1 = list2[0]
            prediction2_2 = list2[1]
            return prediction1_1, prediction1_2, prediction2_1, prediction2_2, lamd, mix_idx, self.acc, gt, idx

class iwildcam_twoTransforms(data.Dataset):
    def __init__(self, ImageNetV2_path, transform=None, transform2=None, without_aug=False):
        self.ImageNetV2_path = ImageNetV2_path
        self.transform1 = transform
        self.transform2 = transform2
        self.file_paths = []
        self.label = []
        dirs = os.listdir(self.ImageNetV2_path)
        dirs.sort()
        label_flag = 0
        self.metadata_path = '/home/liuyc/data/iwildcam_v2.0/metadata.csv'
        self.metadata_pd = pd.read_csv(self.metadata_path) 
        self.test_pd = self.metadata_pd[self.metadata_pd['split']=='id_val']
        self._n_classes = max(self.metadata_pd['y']) + 1
        assert len(np.unique(self.metadata_pd['y'])) == self._n_classes

        for index, row in self.test_pd.iterrows():
            filename = row['filename']
            y = row['y']
            self.label.append(y)
            self.file_paths.append(os.path.join(self.ImageNetV2_path, filename))

    def __len__(self):
        # print(len(self.file_paths))
        # assert(0)
        return len(self.file_paths)

    def __getitem__(self, idx):

        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])

        image = image[:, :, ::-1]

        image2 = self.transform2(image)

        image1 = self.transform1(image)

        return image1, image2, label, idx

class cifar10Predictions_gt(data.Dataset):
    def __init__(self, ImageNetV2_path, prediction_path1, prediction_path2, prediction_path3, type='grey'):
        self.ImageNetV2_path = ImageNetV2_path
        self.prediction_path1 = prediction_path1
        self.prediction_path2 = prediction_path2
        self.prediction_path3 = prediction_path3
        self.type = type
        self.prediction_dic1 = np.load(self.prediction_path1, allow_pickle=True).item()
        self.prediction_dic2 = np.load(self.prediction_path2, allow_pickle=True).item()
        self.prediction_dic3 = np.load(self.prediction_path3, allow_pickle=True).item()
        self.acc = self.prediction_dic1['acc'].item()
        self.keys = self.prediction_dic1.keys()
        
        if '-C' in ImageNetV2_path:
            self.label = np.load(os.path.join(self.ImageNetV2_path, 'labels.npy'))[40000:50000]
        elif '10.1' in ImageNetV2_path:
            self.label = np.load(os.path.join(self.ImageNetV2_path, 'cifar10.1_v4_labels.npy'))
        else:
            self.label = []
            dirs = os.listdir(self.ImageNetV2_path)
            dirs.sort()
            label_flag = 0
            for dir in dirs:
                dir_path = os.path.join(self.ImageNetV2_path, dir)
                if not os.path.isdir(dir_path):
                    continue
                img_names = os.listdir(dir_path)
                for img in img_names:       
                    self.label.append(label_flag)
                label_flag+=1
        
    def __len__(self):
        return len(self.keys)-1

    def __getitem__(self, idx):
        list1 = self.prediction_dic1[str(torch.tensor(idx))]
        list2 = self.prediction_dic2[str(torch.tensor(idx))]
        list3 = self.prediction_dic3[str(torch.tensor(idx))]
        
        # # cifar10 test plugins
        # list1 = self.prediction_dic1[str(idx)]
        # list2 = self.prediction_dic2[str(idx)]
        # list3 = self.prediction_dic3[str(idx)]
        try:
            gt = self.label[idx]
        except:
            print(len(self.label))
            print(idx)
            assert(0)
    
        if self.type == 'grey':
            prediction1_1 = list1[0]
            prediction1_2 = list1[1]
            prediction2_1 = list2[0]
            prediction2_2 = list2[1]
            prediction3_2 = list3[1]
            return prediction1_1, prediction1_2, prediction2_1, prediction2_2, prediction3_2, self.acc, gt, idx
        elif self.type == 'mixup':
            prediction1_1 = list1[0]
            prediction1_2 = list1[1]
            lamd = list1[2]
            mix_idx = list1[3].item()
            prediction2_1 = list2[0]
            prediction2_2 = list2[1]
            return prediction1_1, prediction1_2, prediction2_1, prediction2_2, lamd, mix_idx, self.acc, gt, idx
        

class cifar10_twoTransforms(data.Dataset):
    def __init__(self, ImageNetV2_path, transform=None, transform2=None, without_aug=False):
        self.ImageNetV2_path = ImageNetV2_path
        self.transform1 = transform
        self.transform2 = transform2
        self.file_paths = []
        self.label = []
        dirs = os.listdir(self.ImageNetV2_path)
        dirs.sort()
        label_flag = 0
        for dir in dirs:
            dir_path = os.path.join(self.ImageNetV2_path, dir)
            if not os.path.isdir(dir_path):
                continue
            img_names = os.listdir(dir_path)
            for img in img_names:       
                self.label.append(label_flag)
                img_path = os.path.join(dir_path, img)
                self.file_paths.append(img_path)
            label_flag+=1

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])

        image = image[:, :, ::-1]

        image2 = self.transform2(image)

        image1 = self.transform1(image)

        return image1, image2, label, idx

class cifar10_C_twoTransforms(data.Dataset):
    def __init__(self, ImageNetV2_path, transform=None, transform2=None, without_aug=False):
        self.ImageNetV2_path = ImageNetV2_path
        self.transform1 = transform
        self.transform2 = transform2
        self.file_paths = []
        self.data = np.load(os.path.join(self.ImageNetV2_path,'gaussian_blur.npy'))[20000:30000]
        self.label = np.load(os.path.join(self.ImageNetV2_path, 'labels.npy'))[20000:30000]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        label = self.label[idx]
        image = self.data[idx]


        image2 = self.transform2(image)

        image1 = self.transform1(image)

        return image1, image2, label, idx

class cifar10_v1_twoTransforms(data.Dataset):
    def __init__(self, ImageNetV2_path, transform=None, transform2=None, without_aug=False):
        self.ImageNetV2_path = ImageNetV2_path
        self.transform1 = transform
        self.transform2 = transform2
        self.file_paths = []
        self.data = np.load(os.path.join(self.ImageNetV2_path,'cifar10.1_v4_data.npy'))
        self.label = np.load(os.path.join(self.ImageNetV2_path, 'cifar10.1_v4_labels.npy'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        label = self.label[idx]
        image = self.data[idx]

        image2 = self.transform2(image)

        image1 = self.transform1(image)

        return image1, image2, label, idx
