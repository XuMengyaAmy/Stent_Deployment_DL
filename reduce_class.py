# -*- coding: utf-8 -*-
import re
import os
import sys
import torch
import numpy as np

file_path = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D4/filelist2.txt'
data_dir = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D4/'

def saveFile(classNum, filenum):
    f = open((data_dir + 'filelist' +'.txt'), 'a')
    f.write(filenum + ' ' + classNum)
    f.write('\n')
    f.close()

file = open(file_path, 'r')
while True:
    text = file.readline()
    if not text:
        break
    npynumber = text[0:3]
    label = text[4:5]
    if (label == '4'): 
        label = '3'
    print(npynumber, label)
    saveFile(label, npynumber)
    
