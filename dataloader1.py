# -*- coding: utf-8 -*-
import re
import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, img_list, img_dir, file_path):

        ''' getting train and test list'''
        self.dataset_dir = img_dir
        self.file_list = []
        
        f = open(img_list, 'r')
        while True:
            idx = f.readline().rstrip()
            if not idx: break
            self.file_list.append(idx)
            
        f.close()
        '''------------------------------'''
        ''' getting real class number'''
        self.real_class = []
        f = open(file_path, 'r')
        while True:
            class_label = f.readline().rstrip()
            if not class_label: break
            self.real_class.append(class_label[4:5])
        f.close()
        '''-----------------------------'''

        # ''' store test class label list'''
        # model_label = open("/media/mmlab/dataset/mengya/StentDetectionDataset/re_label.txt", "r+")
        # if test:
        #     for value in self.file_list:
        #         model_label.write(self.real_class[int(value)]+"\n")
        # model_label.close()
        # '''-----------------------------'''

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # _img = Image.open(self.img_anno[index]).convert('RGB')

        _img = np.load(self.dataset_dir+self.file_list[index]+".npy")
        _label = int(self.real_class[int(self.file_list[index])])

        #print(self.dataset_dir+self.file_list[index]+".npy", _label)

        _img = _img.astype(float)/255.0
        _img = _img.transpose(2,0,1)
        #_img.reshape(2,289,110)
        _img = torch.from_numpy(_img).float()     # .float()

        ######## four dimensional data  ####################
        _img = _img.unsqueeze(0)

        _label = np.array(_label)
        _label = torch.from_numpy(_label)

        return _img, _label.item()
