#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 08:39:22 2018

@author: math638
"""

import torch
from torch.utils import data
import os
import PIL.Image as Image
import torchvision.transforms as transforms

class VOCdataset(data.Dataset):
    def __init__(self, root, job='train'):
        list_file = os.path.join(root, "ImageSets/Segmentation/%s.txt"%job)
        img_file_root = os.path.join(root, "JPEGImages")
        label_file_root = os.path.join(root, "SegmentationClass")
        self.list = open(list_file)
        self.files = []
        for l in self.list:
            l=l.strip("\n")
            img_file = os.path.join(img_file_root, "%s.jpg"%l)
            label_file = os.path.join(label_file_root, "%s.png"%l)
            self.files.append({"img": img_file, 
                               "label": label_file
                               })
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data = self.files[index]
        
        img_files = data["img"]
        label_files = data["label"]
        img = Image.open(img_files).convert("RGB")
        img = img.resize((256, 256))
        label = Image.open(label_files).convert("P")
        label = label.resize((256, 256))
        
        img = transforms.ToTensor()(img)
        label = transforms.ToTensor()(label)
        
        return img, label
