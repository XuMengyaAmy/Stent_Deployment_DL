# -*- coding: utf-8 -*-
import re
import os
import sys
import torch
import numpy as np

file_path = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D6/filelist.txt'
data_dir = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D6/'

def saveFile(classNum, filenum):
    f = open((data_dir+classNum+'.txt'), 'a')
    f.write(filenum)
    f.write('\n')
    f.close()

file = open(file_path, 'r')
while True:
    text = file.readline()
    if not text:
        break
    npynumber = text[0:3]
    label = text[4:5]
    if(label =='0' or label== '1' or label=='2' or label== '3' or label== '4'):saveFile(label, npynumber)