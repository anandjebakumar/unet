# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:52:22 2020

@author: Anand Jebakumar
"""


import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_conv1 = self.double_conv(1,64)
        self.down_conv2 = self.maxPool_doubleConv(64,128)
        self.down_conv3 = self.maxPool_doubleConv(128,256)
        self.down_conv4 = self.maxPool_doubleConv(256,512)
        self.down_conv5 = self.maxPool_doubleConv(512,1024)
        
    def forward(self,image):
        # encoder
        x1 = self.down_conv1(image)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)
        x4 = self.down_conv4(x3)
        x5 = self.down_conv5(x4)
        
        # decoder
        x = self.convTranspose(1024,512,x5)
        x = self.addEnc_doubleConv_convTranspose(512,x4,x)
        x = self.addEnc_doubleConv_convTranspose(256,x3,x)
        x = self.addEnc_doubleConv_convTranspose(128,x2,x)
        x = self.addEnc_doubleConv(64,x1,x)
        x = self.outConv(64,2,x)
        return x
    
    def double_conv(self,in_c,out_c):
        conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=3), # default stride=1 and padding=0
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c,out_c,kernel_size=3),
            nn.ReLU(inplace=True)
            )
        return conv
    
    def maxPool_doubleConv(self,in_c,out_c):
        conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            self.double_conv(in_c,out_c)
            )
        return conv
    
    def crop_tensor(self,tensor,target_tensor):
        tensor_size = tensor.shape[2]
        target_tensor_size = target_tensor.shape[2]
        delta = tensor_size-target_tensor_size
        delta = delta // 2
        return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]
    
    def convTranspose(self,in_c,out_c,x):
        return nn.ConvTranspose2d(in_c,out_c,kernel_size=2,stride=2)(x)
    
    def addEnc_doubleConv(self,n_ch,x_encoder,x):
        y = self.crop_tensor(x_encoder,x)
        x = torch.cat([y,x],dim=1)
        x = self.double_conv(n_ch*2,n_ch)(x)
        return x
    
    def addEnc_doubleConv_convTranspose(self,n_ch,x_encoder,x):
        x = self.addEnc_doubleConv(n_ch,x_encoder,x)
        x = self.convTranspose(n_ch,n_ch//2,x)
        return x

    def outConv(self,in_c,out_c,x):
        return nn.Conv2d(in_c,out_c,kernel_size=1)(x)
    
unet = UNet()
image = torch.randn([1,1,572,572])
out = unet(image)
print(out)
print(out.shape)