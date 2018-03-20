#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 14:11:31 2018

@author: math638
"""

import torch.utils.data as data
from dataset import VOCdataset
from model import Seg
#import torch.optim as optim
import torch.nn as nn
import torch
from resnet import resnet50
from torch.autograd.variable import Variable
import torch.nn.functional as F

cnn = resnet50(pretrained = True)
cnn = cnn.cuda()
#print(cnn)

weight = torch.ones(22)
weight[21] = 0
seg = Seg()
seg = seg.cuda()

def run_model():
    loader = data.DataLoader(VOCdataset("/home/math638/Documents/VOCdevkit/VOC2012"),batch_size = 5)
    optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr = 1e-4)
    optimizer_seg = torch.optim.Adam(seg.parameters(), lr = 1e-4)
    loss = nn.NLLLoss2d(weight.cuda(), size_average=True)
    
    for i in range(20):
        for j, (image, label) in enumerate(loader):
            image = image.cuda()
            label = label.cuda()
            image = Variable(image)
#            image = image.unsqueeze(0)
#            image = image.type(torch.cuda.FloatTensor)
            label = Variable(label)
            label = label.type(torch.cuda.LongTensor)
            label = torch.squeeze(label)
#            label = torch.unsqueeze(label, 0)
            feats = cnn(image)
#            feats = Variable(feats)
            prediction = seg(feats)
            
            seg.zero_grad()
            cnn.zero_grad()
            
#            print(prediction.size())
#            print(label.size())
            l = loss(F.log_softmax(prediction), label)
            
            l.backward()
            optimizer_cnn.step()
            optimizer_seg.step()
            
            print("epoch %d  step %d  loss=%f"%(i,j,l.data.cpu()[0]))
    
    torch.save(cnn.state_dict(), 'cnn_params.pkl')
    torch.save(seg.state_dict(), 'seg_params.pkl')
 