# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets #手写数据集要用到
from sklearn.manifold import TSNE
import pylab

#该函数是关键，需要根据自己的数据加以修改，将图片存到一个np.array里面，并且制作标签
#因为是两类数据，所以我分别用0,1来表示
def get_data(Input_path, file_path, img_list): #Input_path为你自己原始数据存储路径，我的路径就是上面的'./Images'
    # count = len(open(img_list,'rU').readlines())
    # print(count)

    # Image_names=os.listdir(Input_path) #获取目录下所有图片名称列表

    # #为前500个分配标签1，后500分配0
    # for k in range(500):
    #     label[k]=1
     
    
    file_list = []       
    f = open(img_list, 'r')
    while True:
        idx = f.readline().rstrip()
        if not idx: break
        file_list.append(idx)                 # file_list stores all the npy name in training dataset or in testing dataset        
    f.close()

    data=np.zeros((len(file_list),31*34*34)) #初始化一个np.array数组用于存数据
    label=np.zeros((len(file_list),)) #初始化一个np.array数组用于存数据

    ''' getting real class number'''
    real_class = []
    f = open(file_path, 'r')
    while True:
        class_label = f.readline().rstrip()
        if not class_label: break
        real_class.append(int(class_label[4:5]))     # real_class stores all the labels of all data (train and test)
    f.close()
    # label = np.array(real_class)
    # print(label)
    # print(label.shape)

    #读取并存储图片数据，原图为rgb三通道，而且大小不一，先灰度化，再resize成200x200固定大小
    for i in range(len(file_list)):
        # image_path=os.path.join(Input_path,Image_names[i])
        image_path=os.path.join(Input_path,file_list[i]+".npy")
        img=np.load(image_path)
        img=img.reshape(1,31*34*34)
        data[i]=img
        n_samples, n_features = data.shape
        # class_number= real_class[int(file_list[i])]
        # label[i] = class_number
        class_number= int(real_class[int(file_list[i])])
        label[i] = class_number

    print(data.shape)
    print(label.shape)
    label = label.astype(int)    # change 0.0 ro 0, This code is necessary
    print(label)
    return data, label, n_samples, n_features

'''
下面的两个函数，
一个定义了二维数据，一个定义了3维数据的可视化
不作详解，也无需再修改感兴趣可以了解matplotlib的常见用法
'''
def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
def plot_embedding_3D(data,label,title): 
    x_min, x_max = np.min(data,axis=0), np.max(data,axis=0) 
    data = (data- x_min) / (x_max - x_min) 
    fig = plt.figure()
    ax = plt.figure().add_subplot(111,projection='3d') 
    for i in range(data.shape[0]): 
        ax.text(data[i, 0], data[i, 1], data[i,2],str(label[i]), color=plt.cm.Set1(label[i]),fontdict={'weight': 'bold', 'size': 9}) 
    return fig

#主函数
def main():
    Input_path  = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D6/Data/'
    file_path = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D6/filelist.txt'
    img_list = "/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D6/train.txt"   # can change train to 0, 1, 2, 3, 4, test 

    data, label, n_samples, n_features = get_data(Input_path, file_path, img_list) #根据自己的路径合理更改
    print('Begining......') #时间会较长，所有处理完毕后给出finished提示
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0) #调用TSNE
    result_2D = tsne_2D.fit_transform(data)
    tsne_3D = TSNE(n_components=3, init='pca', random_state=0)
    result_3D = tsne_3D.fit_transform(data)
    print('Finished......')
    #调用上面的两个函数进行可视化
    fig1 = plot_embedding_2D(result_2D, label,'t-SNE')
    plt.show(fig1)
    pylab.show()
    fig2 = plot_embedding_3D(result_3D, label,'t-SNE')
    plt.show(fig2)
    pylab.show()

if __name__ == '__main__':
    main()