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

#from unet_parts import *

# import torchvision.models as models
from torchsummary import summary



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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

    plt.savefig('/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D8/loss_curve.png', format='png')    
    plt.show()
    
    pylab.show()


# train data path
train_xdir = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D8/Data/'
train_ypath = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D8/filelist.txt'
train_data_list = "/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D8/train.txt"

# test data path
test_xdir = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D8/Data/'
test_ypath = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D8/filelist.txt'
test_data_list= "/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D8/test.txt"

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

'''
# Define model for 4 dimensional data 
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 6, 5),  # in_channels, out_channels, kernel_size 
            nn.ReLU(),
            nn.MaxPool3d(2, 2), # kernel_size, stride
            nn.Conv3d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool3d(2, 2)
        )
        self.fc = nn.Sequential(

            # nn.Linear(16*203*232, 120), # num_inputs, num_outputs
            nn.Linear(16 * 5 * 5 * 4, 200),  # num_inputs, num_outputs
            nn.ReLU(),
            nn.Linear(200, 120),
            nn.ReLU(),
            nn.Linear(120, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 4)     # Before: 4 classes in total, Now 5 classes in total
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))   # do the flatten 
        return output
'''


'''
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
'''
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


############################ Complicated network #################################################
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 10)
#         self.down1 = Down(10, 20)
#         self.down2 = Down(20, 30)
#         self.down3 = Down(30, 40)
#         #self.down4 = Down(256, 512)
        
        
#         self.linear1 = nn.Linear(86700, 256) #flatten layer size of down1
#         self.linear2 = nn.Linear(13440, 256) #flatten layer size of down2
#         self.linear3 = nn.Linear(1920, 256) #flatten layer size of down3
#         #self.linear4 = nn.Linear(, 256) #flatten layer size of down4
#         self.linear5 = nn.Linear(768, 128)
#         self.linear6 = nn.Linear(128, 64)
#         self.linear7 = nn.Linear(64, n_classes)     # classes number 

#         self.activation1 = nn.ReLU()
#         self.activation2 = nn.ReLU()
#         self.activation3 = nn.ReLU()

#     def forward(self, x):
#         # print(x.size())
#         x = self.inc(x)
#         x = self.down1(x)
#         x1 = self.linear1(x.view(x.shape[0], -1))
#         # print("first1")
#         # print(x.size())
#         x = self.down2(x)
#         x2 = self.linear2(x.view(x.shape[0], -1))
#         # print("first2")
#         # print(x.size())
#         x = self.down3(x)
#         x3 = self.linear3(x.view(x.shape[0], -1))
#         # print("first3")

#         full_fea_x = torch.cat((x1,x2,x3), 1)
#         # print(full_fea_x.size())
#         full_fea_x = self.linear5(full_fea_x)
#         full_fea_x = self.activation1(full_fea_x)
#         full_fea_x = self.linear6 (full_fea_x)
#         full_fea_x = self.activation2(full_fea_x)
#         full_fea_x = self.linear7(full_fea_x)
#         return full_fea_x

############################ Simpler network #################################################
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 10)
        self.down1 = Down(10, 20)
        # self.down2 = Down(20, 30)
        # self.down3 = Down(30, 40)
        # #self.down4 = Down(256, 512)
        
        
        self.linear1 = nn.Linear(86700, 1024) #flatten layer size of down1
        # self.linear2 = nn.Linear(13440, 256) #flatten layer size of down2
        # self.linear3 = nn.Linear(1920, 256) #flatten layer size of down3
        # #self.linear4 = nn.Linear(, 256) #flatten layer size of down4
        self.linear5 = nn.Linear(1024, 512)
        self.linear6 = nn.Linear(512, 64)
        self.linear7 = nn.Linear(64, n_classes)    # classes number 

        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.ReLU()

    def forward(self, x):
        # print(x.size())
        x = self.inc(x)
        x = self.down1(x)
        x = self.linear1(x.view(x.shape[0], -1))
        # print("first1")
        # print(x.size())
        # x = self.down2(x)
        # x2 = self.linear2(x.view(x.shape[0], -1))
        # # print("first2")
        # # print(x.size())
        # x = self.down3(x)
        # x3 = self.linear3(x.view(x.shape[0], -1))
        # print("first3")

        # full_fea_x = torch.cat((x1,x2,x3), 1)
        # print(full_fea_x.size())
        full_fea_x = self.linear5(x)
        full_fea_x = self.activation1(full_fea_x)
        full_fea_x = self.linear6 (full_fea_x)
        full_fea_x = self.activation2(full_fea_x)
        full_fea_x = self.linear7(full_fea_x)
        return full_fea_x


global net
# net = LeNet()
# net = UNet(1, 3)  # classes number 
net = UNet(1, 4)  # classes number 

net = net.to(device)
summary(net, (1, 31, 34, 34))
print(net)


# ######## Test the shape of con block #######
# X = torch.rand(1, 31, 34, 34)
# X = X.to(device)

# for name, blk in net.named_children():
#     X = blk(X)
#     print(name, 'output shape', X.shape)






pre_label = open("/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D8/pre_label.txt", "w+")
model_label = open("/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D8/re_label.txt", "w+")



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


            # pre_label.close()  #  have some error
 

            # print(y_pred)
            # print(y_real)
            # print('-'*10)

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
    
    ##########  find the best epoch  ##########
    # Best_Dice = 0
    # Best_epoch=0
    # avg_dice = []
    ###########################################

    #net = net.to(device)
    #print("training on ", device)
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
        
        
        ##########  find the best epoch  ##########
        # if np.mean(avg_dice) > Best_Dice:
        #     Best_Dice = np.mean(avg_dice)
        #     Best_epoch = epochs
        #3############################################
        

        
        PATH = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D8/checkpoint/'
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        # 'ckpt_dir': '/media/mmlab/data/mengya/miccai_2018_SurgicalScene/trained_model'
        # torch.save(net.state_dict(), PATH)
        snapshot_name = 'epoch_' + str(epoch)
        torch.save(net.state_dict(), os.path.join(PATH, snapshot_name + '.pt'))

    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])  

# lr = 0.00005
# w_d = 0.05 
learning_rate = 0.00005
weight_decay = 0.4
num_epochs = 150

train_ch5(train_iter, test_iter, batch_size, device, learning_rate, weight_decay, num_epochs)

