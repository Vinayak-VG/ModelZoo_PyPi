import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class U_Net(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=np.sqrt(2/(3*3*64)))
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=np.sqrt(2/(3*3*64)))
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=np.sqrt(2/(3*3*128)))
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv4.weight, mean=0.0, std=np.sqrt(2/(3*3*128)))
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv5.weight, mean=0.0, std=np.sqrt(2/(3*3*256)))
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv6.weight, mean=0.0, std=np.sqrt(2/(3*3*256)))
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv7.weight, mean=0.0, std=np.sqrt(2/(3*3*512)))
        self.conv8 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv8.weight, mean=0.0, std=np.sqrt(2/(3*3*512)))
        self.maxpool4 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.conv9 = nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv9.weight, mean=0.0, std=np.sqrt(2/(3*3*1024)))
        self.conv10 = nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=np.sqrt(2/(3*3*1024)))
        self.upconv1 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2)
        nn.init.normal_(self.upconv1.weight, mean=0.0, std=np.sqrt(2/(2*2*512)))
        self.conv11 = nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv11.weight, mean=0.0, std=np.sqrt(2/(3*3*512)))
        self.conv12 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv12.weight, mean=0.0, std=np.sqrt(2/(3*3*512)))
        self.upconv2 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2)
        nn.init.normal_(self.upconv2.weight, mean=0.0, std=np.sqrt(2/(2*2*256)))
        self.conv13 = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv13.weight, mean=0.0, std=np.sqrt(2/(3*3*256)))
        self.conv14 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv14.weight, mean=0.0, std=np.sqrt(2/(3*3*256)))
        self.upconv3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2)
        nn.init.normal_(self.upconv3.weight, mean=0.0, std=np.sqrt(2/(2*2*128)))
        self.conv15 = nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv15.weight, mean=0.0, std=np.sqrt(2/(3*3*128)))
        self.conv16 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv16.weight, mean=0.0, std=np.sqrt(2/(3*3*128)))
        self.upconv4 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2)
        nn.init.normal_(self.upconv4.weight, mean=0.0, std=np.sqrt(2/(2*2*64)))
        self.conv17 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv17.weight, mean=0.0, std=np.sqrt(2/(3*3*64)))
        self.conv18 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        nn.init.normal_(self.conv18.weight, mean=0.0, std=np.sqrt(2/(3*3*64)))
        self.conv19 = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
        nn.init.normal_(self.conv19.weight, mean=0.0, std=np.sqrt(2/(1*1*1)))
        self.relu = nn.ReLU()
        self.gn1 = nn.GroupNorm(16, 64)
        self.gn2 = nn.GroupNorm(16, 128)
        self.gn3 = nn.GroupNorm(16, 256)
        self.gn4 = nn.GroupNorm(16, 512)
        self.gn5 = nn.GroupNorm(16, 1024)
        self.dropout = nn.Dropout(0.5)
        self.dropout1 = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gn1(x)
        x = self.relu(x)
        out1 = x
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.gn2(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.gn2(x)
        x = self.relu(x)
        out2 = x
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.gn3(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.gn3(x)
        x = self.relu(x)
        out3 = x
        x = self.maxpool3(x)
        x = self.dropout(x)
        x = self.conv7(x)
        x = self.gn4(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.gn4(x)
        x = self.relu(x)
        out4 = x
        x = self.maxpool4(x)
        x = self.dropout(x)
        x = self.conv9(x)
        x = self.gn5(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.gn5(x)
        x = self.relu(x)
        x = self.upconv1(x)
        x = torch.cat((x, out4), 1)
        x = self.dropout(x)
        x = self.conv11(x)
        x = self.gn4(x)
        x = self.relu(x)
        x = self.conv12(x)
        x = self.gn4(x)
        x = self.relu(x)
        x = self.upconv2(x)
        x = torch.cat((x, out3), 1)
        x = self.dropout(x)
        x = self.conv13(x)
        x = self.gn3(x)
        x = self.relu(x)
        x = self.conv14(x)
        x = self.gn3(x)
        x = self.relu(x)
        x = self.upconv3(x)
        x = torch.cat((x, out2), 1)
        x = self.dropout(x)
        x = self.conv15(x)
        x = self.gn2(x)
        x = self.relu(x)
        x = self.conv16(x)
        x = self.gn2(x)
        x = self.relu(x)
        x = self.upconv4(x)
        x = torch.cat((x, out1), 1)
        x = self.dropout(x)
        x = self.conv17(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv18(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv19(x)

        return x