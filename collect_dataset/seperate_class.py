# -*- coding: utf-8 -*-
import re
import os
import sys
import torch
import numpy as np

file_path = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/filelist.txt'
data_dir = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D1/'

def saveFile(classNum, filenum):
    f = open((data_dir+classNum+'.txt'), 'a')
    f.write(filenum)
    f.write('\n')
    f.close()

raw_label = []
file = open(file_path, 'r')
while True:
    text = file.readline()
    if not text:
        break
    # result = [int(d) for d in re.findall(r'-?\d+', text)]
    npynumber = text[0:3]
    label = text[4:5]
    file_path = data_dir + npynumber + '.npy'
    if(label =='0' or label== '1' or label=='2' or label=='3'):saveFile(label, npynumber)




#     raw_label.append(text[4:5])
# print(raw_label)
# print(len(raw_label))

# a, b, c, d = 0, 0, 0, 0

# for item in raw_label:
#     if item == '0':
#         a += 1
#     if item == '1':
#         b += 1
#     if item == '2':
#         c += 1
#     if item == '3':
#         d += 1
# print('class 0: %d, class 1: %d, class 2: %d, class 3: %d' % (a, b, c, d))

# file_path = data_dir +  text[0:3].npy

# def save(html, path):
#     '''
#     以文件形式保存数据
#     :param html: 要保存的数据
#     :param path: 要保存数据的路径
#     :return:
#     '''
#     # 判断目录是否存在
#     if not os.path.exists(os.path.split(path)[0]):
#         # 目录不存在创建，makedirs可以创建多级目录
#         os.makedirs(os.path.split(path)[0])
#     try:
#         # 保存数据到文件
#         with open(path, 'wb') as f:
#             f.write(html.encode('utf8'))
#         print('保存成功')
#     except Exception as e:
#         print('保存失败', e)


# if __name__ == "__main__":
#     html = '数据'  # 要保存的数据
#     path = 'D:/a/b/1.txt'  # 设置路径，也可设为相对路径
#     save(html, path)