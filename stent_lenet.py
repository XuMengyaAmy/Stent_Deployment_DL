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



##################################################################
'''
# Define LENET model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(34, 6, 5),  # in_channels, out_channels, kernel_size 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(

            # nn.Linear(16*203*232, 120), # num_inputs, num_outputs
            nn.Linear(16 * 5 * 4, 120),  # num_inputs, num_outputs
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 4)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

'''
##################################################################

# Define LENET model for 4 dimensional data 
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
        output = self.fc(feature.view(img.shape[0], -1))
        return output

##################################################################
'''
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小小卷积窗口口,使用用填充为2来使得输入入与输出的高高和宽一一致,且增大大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层,且使用用更更小小的卷积窗口口。除了了最后的卷积层外,进一一步
            # 增大大了了输出通道数。
            # 前两个卷积层后不不使用用池化层来减小小输入入的高高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        # 这里里里全连接层的输出个数比比LeNet中的大大数倍。使用用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),   ####
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里里里使用用Fashion-MNIST,所以用用类别数为10,而而非非论文文中的1000
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
'''
##################################################################


global net
net = LeNet()
# net = AlexNet()

net = net.to(device)
print(net)


'''
######## Test the shape of con block #######
X = torch.rand(1, 1, 34, 31)
X = X.to(device)

for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape', X.shape)
'''

#     # print(X.view(16*203*232, -1).shape)



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


################### Consider Dropout Layer ################
'''
def evaluate_accuracy(data_iter, device):
    global net
    if device is None and isinstance(net, torch.nn.Module):
       device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    m = 0
    batch_count = 0
    loss = torch.nn.CrossEntropyLoss()
    test_l_sum = 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
            
                
                y_hat = net(X.to(device))
                y = y.to(device)
                l = loss(y_hat, y)

                y_pred = y_hat.argmax(dim=1).cpu().numpy()

                y_real = y.cpu().numpy()   # if y = , the original y will be covered, so change into another name
                    
                for item in y_pred: 
                    # print(item)
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

            else:
                if ('is_training' in net.__code__.co_varnames):

                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()

            n += y.shape[0]
            batch_count += 1
            test_l_sum += l.cpu().item()  #
        test_loss = test_l_sum/batch_count
    return (acc_sum /n, test_loss)
'''

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
        

        
        PATH = '/media/mmlab/dataset/mengya/StentDetectionDataset/Dataset/D5/checkpoint/'
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        # 'ckpt_dir': '/media/mmlab/data/mengya/miccai_2018_SurgicalScene/trained_model'
        # torch.save(net.state_dict(), PATH)
        snapshot_name = 'epoch_' + str(epoch)
        torch.save(net.state_dict(), os.path.join(PATH, snapshot_name + '.pt'))

    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])  

# lr = 0.00005
# w_d = 0.05 
learning_rate = 0.00002
weight_decay = 0.012
num_epochs = 150

train_ch5(train_iter, test_iter, batch_size, device, learning_rate, weight_decay, num_epochs)

