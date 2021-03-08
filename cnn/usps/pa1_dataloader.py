#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:50:34 2021

@author: chashi
"""
import numpy as np
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def get_dataset(filepath):
    samples = []
    labels = []
    with open(filepath, "r") as file1:
# =============================================================================
#     with open("zip_test.txt", "r") as file1:
# =============================================================================
        for line in file1.readlines():
            f_list = [float(i) for i in line.split() if i!='\n']
            label = np.array([f_list[0]])
            sample = np.array([f_list[1:]])
            samples.append(sample)
            labels.append(label)
    return samples, labels


def load( train, batch_size=16, shuffle=True):
    filepath = "zip_train.txt"
    if train == False:
        filepath = "zip_test.txt"
    samples, labels = get_dataset(filepath)
    tensor_x = torch.Tensor(samples)
    tensor_x = tensor_x.view(-1, 1, 16, 16)
    tensor_y = torch.Tensor(labels)
    tensor_y = tensor_y.view(-1)
    print(tensor_x.shape, " ", tensor_y.shape)
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

if __name__ == "__main__":
    data_loader = load(train = True, batch_size = 16)
    
    di = iter(data_loader)
    
    s, i = di.next() 
    images, labels = di.next()
    images = images.numpy()
    images = images.reshape(16, 1 , 16, 16)
    
    plt.imshow(images[5].squeeze(), cmap='Greys_r')
    plt.title(str(labels[5].numpy()))    
