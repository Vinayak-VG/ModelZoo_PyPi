from modelzoo_iitm import models
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
import math
from torch.autograd import Variable

def test_modelzoo_no_params():
    assert models() == "No Models loaded"

def test_modelzoo_with_params():
    assert models("PointNet") == "PointNet model loaded successfully"

# def test_UNet():
#     assert 