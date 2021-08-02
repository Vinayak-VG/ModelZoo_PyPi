import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from UNet import *
from UNetPP import *
from PointNet import *
from StackGAN import *
from CharCNN import *
from PSPNet import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def models(name = None):
    if name is None:
        return "No Models loaded"
    else:
        return f"{name} model loaded successfully"

def UNet(num_channels = 1, pretrained = False):        # num_channels = 1 if grayscale image else num_channels = 3 if color image
    if pretrained == True:
        model = U_Net(num_channels)
        url = ""
        model.load_state_dict(model_zoo.load_url(url))
    else:
        model = U_Net(num_channels)
        return model

def UNetPP(num_channels = 1):      # num_channels = 1 if grayscale image else num_channels = 3 if color image
    model = UNet_PP(num_channels, DenseBlock, BasicDownBlock, BasicUpBlock)
    return model

def PointNet():
    model = Point_Net(T_Net)
    model = model.apply(weights_init)
    return model

def CharCNN():
    model = Char_CNN()
    return model

# def KimCNN(vocab_size = 20000, embedding_dim = 300, n_filters = 100, filter_sizes = [3, 4, 5], output_dim = 1, dropout = 0.2, pad_idx = 0):
#     model = Kim_CNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx)
#     return model

def StackGAN(name = "None"):
    if name == "None":
        return "Please specify the name of Model"
    elif name == 'StageI_Gen':
        model = StageI_GAN_Gen(Conditioning_Augmentation_StageI)
        model = model.apply(weights_init)
        return model
    elif name == 'StageI_Dis':
        model = StageI_GAN_Dis(DownSample1)
        model = model.apply(weights_init)
        return model
    elif name == 'StageII_Gen':
        model = StageII_GAN_Gen(DownSample2, ResidualBlock, UpSampling2, Conditioning_Augmentation_StageII)
        model = model.apply(weights_init)
        return model
    elif name == 'StageII_Dis':
        model = StageII_GAN_Dis(DownSample3)
        model = model.apply(weights_init)
        return model

def PSPNet():
    model = PSP_Net()
    return model

