# -*- coding: utf-8 -*-
import re
import os
import sys
import torch
import numpy as np

file_path = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D6/4.txt'
data_dir = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D6/'

def saveFile(rawFileList,fileList, listType):
    f = open((data_dir+listType+'.txt'), 'a')
    for item in fileList:
        f.write(rawFileList[item])
    f.close()

rawFileList = []
file = open(file_path, 'r')
while True:
    text = file.readline()
    rawFileList.append(text)
    if not text: break

total_idx = len(rawFileList)
idx = np.random.permutation(total_idx)   # shuffle imgs
trian_idx = idx[:int(0.7*total_idx)]
test_idx = idx[int(0.7*total_idx):]
print(int(0.7*total_idx))
saveFile(rawFileList, trian_idx, 'train')
saveFile(rawFileList,test_idx, 'test')


