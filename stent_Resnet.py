import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# -*- coding: utf-8 -*-
import time
import torch
from torch import nn, optim
import sys
sys.path.append("..")
#import d2lzh_pytorch as d2l

# import dataloader1
import dataloader1
import os
import numpy as np


import torch
import torchvision
import torchvision.transforms as transforms
import sys
from IPython import display
from matplotlib import pyplot as plt
import pylab
from torch import nn
import time
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize



def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)

    plt.savefig('/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D5/loss_curve.png', format='png')    
    plt.show()
    
    # pylab.show()


# train data path
train_xdir = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D5/Data/'
train_ypath = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D5/filelist.txt'
train_data_list = "/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D5/train.txt"

# test data path
test_xdir = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D5/Data/'
test_ypath = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D5/filelist.txt'
test_data_list= "/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D5/test.txt"
#dataloader
train_dataset = dataloader1.Dataset(img_list = train_data_list, img_dir=train_xdir, file_path=train_ypath)
test_dataset = dataloader1.Dataset(img_list = test_data_list, img_dir=test_xdir, file_path=test_ypath)


feature, label = train_dataset[4]
print(feature.shape, label)



batch_size = 40

if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 2

train_iter = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)



###########################################################################################
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


net = nn.Sequential(
    nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm3d(64),
    nn.ReLU(),
    nn.MaxPool3d(kernel_size=3, stride=2, padding=1))

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一一个模块的通道数同输入入通道数一一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))

net.add_module("global_avg_pool", d2l.GlobalAvgPool3d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, 5)))

# X = torch.rand((1, 1, 31, 34, 34))
# for name, layer in net.named_children():
#     X = layer(X)
#     print(name, ' output shape:\t', X.shape)

########################################################################################################



global net
# net = LeNet()
# # net = AlexNet()

net = net.to(device)
print(net)


pre_label = open("/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D5/pre_label.txt", "w+")
model_label = open("/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D5/re_label.txt", "w+")



# train the model
def evaluate_accuracy(data_iter, device, epoch):
    global net
    #if device is None and isinstance(net, torch.nn.Module):
    #    device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    m = 0
    batch_count = 0
    loss = torch.nn.CrossEntropyLoss()
    test_l_sum = 0
    
    pre_label.write("epoch: "+str(epoch) +"\n")
    model_label.write("epoch: "+str(epoch) +"\n")
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() 
                
            y_hat = net(X.to(device))
            y = y.to(device)
            l = loss(y_hat, y)

            y_pred = y_hat.argmax(dim=1).cpu().numpy()

            y_real = y.cpu().numpy()   # if y = , the original y will be covered, so change into another name
                
            for item in y_pred:
                pre_label.write(item.astype(str) +"\n")    #.astype(str) is necessary
            
            # model_label.write(y_real.astype(str) +"\n")

            for item in y_real: 
                # print(item)
                model_label.write(item.astype(str) +"\n")    #.astype(str) is necessary


          

            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            
            n += y.shape[0]
            batch_count += 1
            test_l_sum += l.cpu().item()  #
        test_loss = test_l_sum/batch_count
    return (acc_sum /n, test_loss)




def train_ch5(train_iter, test_iter, batch_size, device, lr, w_d, num_epochs):
    global net
    
    train_ls = []
    test_ls = []
    
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = w_d)
        net.train()
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:

            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()  #
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1  #
        test_acc, test_l = evaluate_accuracy(test_iter,device, epoch)

        train_loss = train_l_sum / batch_count  #
        train_ls.append(train_loss)
        test_ls.append(test_l)
        print('epoch %d, train_loss %.4f, test_loss %.4f, overfit %.4f,train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_loss, test_l, test_l-train_loss, train_acc_sum / n, test_acc, time.time() - start))
        
        
        PATH = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D5/checkpoint/'
        if not os.path.exists(PATH):
            os.makedirs(PATH)
      
        snapshot_name = 'epoch_' + str(epoch)
        torch.save(net.state_dict(), os.path.join(PATH, snapshot_name + '.pt'))

    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])  


learning_rate = 0.00002
weight_decay = 0.012
num_epochs = 150

train_ch5(train_iter, test_iter, batch_size, device, learning_rate, weight_decay, num_epochs)

