# -*- coding: utf-8 -*-
import re
import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# PIXEL_MEAN = [0.4634645879268646, 0.35329848527908325, 0.3741738796234131]
# PIXEL_MEAN = np.array(PIXEL_MEAN, dtype=np.float32)


class Dataset(Dataset):
    def __init__(self, img_dir, file_path, train=True, test=False):

        # imgname = os.listdir(img_dir)
        # imgname.sort()

        imgs = [os.path.join(img_dir, imgname) for imgname in os.listdir(img_dir)] 
        # imgs = [os.path.join(img_dir, imgname)
        imgs_num = len(imgs)

       
        imgs = np.random.permutation(imgs)   # shuffle imgs

        # Train : Validation = 7 : 3
        if train:
            self.imgs = imgs[:int(0.7*imgs_num)]
            print(self.imgs)
        elif test:
            self.imgs = imgs[int(0.7*imgs_num):]  
            print('-'*100)
            print(self.imgs)  

        # file_img = open(img_dir, 'r')
        re_label = open("/media/mmlab/dataset/mengya/StentDetectionDataset/re_label.txt", "r+")

        self.img_anno = {}

        npy_index = []

        for idx, item in enumerate(self.imgs):
            self.img_anno[idx] = item
            npy_index.append(int(item[61:64]))


        self.raw_label = []
        self.file_path = file_path
        file = open(self.file_path, 'r')
        while True:
            text = file.readline()
            if not text:
                break
            # result = [int(d) for d in re.findall(r'-?\d+', text)]
            npy = text[0:3]
            self.raw_label.append(text[4:5])
        

        self.real_label = []
        #print(npy_index)
        #print(len(self.raw_label))
        for value in  npy_index:
            #print(value)
            self.real_label.append(self.raw_label[value])
            if test:
                re_label.write(self.raw_label[value]+"\n")
        re_label.close()


    def __len__(self):
        return len(self.img_anno)

    def __getitem__(self, index):
        # _img = Image.open(self.img_anno[index]).convert('RGB')
        _img = np.load(self.img_anno[index])
        _label = int(self.real_label[index])

        # print(_img.dtype)
        #_img = np.array(_img, dtype=np.uint8)
        # _img -= PIXEL_MEAN
        # _img = torch.from_numpy(_img.transpose(2, 0, 1)).float()
        _img = torch.from_numpy(_img).float()     # .float()
        #
        _label = np.array(_label)
        _label = torch.from_numpy(_label)

        # ran_img = torch.rand(3, 826, 942)

        # return _img, _label[0]   # 2D heat map
        return _img, _label.item()




