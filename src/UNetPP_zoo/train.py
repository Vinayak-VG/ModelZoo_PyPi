import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
from torchvision.utils import save_image
import glob
import math 
from sklearn.model_selection import train_test_split
import gc
import time
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from UNetPP_zoo.model import *
from tqdm import trange

epoch_train_losses = []              # Defining an empty list to store the epoch losses
epoch_val_losses = []             
accu_train_epoch = []                # Defining an empty list to store the accuracy per epoch
accu_val_epoch = []

model = UNet_PP(num_channels = 1, denseblock = DenseBlock, basicdownblock = BasicDownBlock, basicupblock = BasicUpBlock)

def make_tensor(tensor):
      if torch.cuda.is_available():
        return torch.cuda.FloatTensor(tensor)
      else:
        return torch.FloatTensor(tensor)

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy(input, target)
        smooth = 1e-5
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice             # We are giving more weightage to Dice Loss

batch_size = 1
epochs = 10
lr = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss = BCEDiceLoss()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr)


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def train_UNetPP(model, dataset, optimizer, criterion, device):

    train_loss_batch = []
    accu_train_batch = []
    model.train()
    for idx,(images, labels) in enumerate(dataset):
        images = images.to(device)
        labels = labels.to(device)

        #Forward Pass
        output = model(make_tensor(images))
        output = torch.clip(output, 0.0025, 0.9975)            # I am clipping the output because if it becomes 0 or 1 then there is a chance that loss function can explode
        labels = torch.round(labels)
        train_loss = criterion(output, labels)
        train_loss_batch.append(train_loss)
        output = torch.round(output)
        acc = iou_score(output, labels)
        accu_train_batch.append(acc)
        print(f"Batch: {idx + 1}   Train Loss: {train_loss:.5f}   Accuracy: {acc:.5f}")
        # Backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    epoch_train_losses.append(sum(train_loss_batch)/len(dataset))
    accu_train_epoch.append(sum(accu_train_batch)/len(dataset))
    print(f"Train Epoch Loss: {(sum(train_loss_batch)/len(dataset)):.5f}   Train Epoch Accuracy: {(sum(accu_train_batch)/len(dataset)):.5f}")

def eval_UNetPP(model, dataset, criterion, device):

    val_loss_batch = []
    accu_val_batch = []
    model.eval()
    for idx,(images, labels) in enumerate(dataset):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            #Forward Pass
            output = model(make_tensor(images))
            torch.clip(output, 0.0025, 0.9975)
            # Loss
            val_loss = criterion(output, labels)
            val_loss_batch.append(val_loss)
            output = torch.round(output)
            acc = iou_score(output, labels)
            accu_val_batch.append(acc)
    epoch_val_losses.append((sum(val_loss_batch))/len(dataset))
    accu_val_epoch.append((sum(accu_val_batch))/len(dataset))
    print(f"Val Epoch Loss: {((sum(val_loss_batch))/len(dataset)):.5f}   Val Epoch Accuracy: {((sum(accu_val_batch))/len(dataset)):.5f}")