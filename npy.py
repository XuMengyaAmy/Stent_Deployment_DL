# -*- coding: utf-8 -*-
import re
import os
import sys
import torch
import numpy as np

file_path = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D4/filelist.txt'

file = open(file_path, 'r')
while True:
    text = file.readline()
    if not text:
        break
    npynumber = text[0:3]
    label = text[4:5]
    if(label=='2'):
        re.sub('2','1', text)
        print("replaced")
        