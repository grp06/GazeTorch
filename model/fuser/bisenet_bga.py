'''
The Bilateral Guided Aggregation (BGA) layer from
https://github.com/CoinCheung/BiSeNet/blob/master/lib/models/bisenetv2.py

ResNet-18: c_L = 128, c_H = 512
ResNet-50: c_L = 512, c_H = 2048

Zheng(2023-07-24): change the output of BGA layer from 'x_gap, out' to 'out' as we don't use 'x_gap' for now;
then use 'all' to get the BGA layer's output, e.g.:
    head_seg:
        module_input:
        fuser: all

'''

import torch
import torch.nn as nn
from model.module_base import Module_base
from tools.torch_ops import ConvBNReLU, UpSample

class CEBlock(nn.Module):

    def __init__(self, c=512):
        super(CEBlock, self).__init__()
        # self.bn = nn.BatchNorm2d(c)
        self.conv_gap = self.left1 = nn.Sequential(
            nn.Conv2d(
                c, c, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.LayerNorm([c,1,1]),
        )
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(c, c, 3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        # feat = self.bn(feat)
        feat_gap = self.conv_gap(feat)
        feat = feat_gap + x
        feat = self.conv_last(feat)
        return feat, feat_gap

class BGALayer(Module_base):
    def __init__(self, name, input_shape, c_L=512, c_H=2048, c_hidden=128):
        Module_base.__init__(self, name=name, input_shape=input_shape)
        self.c_L = c_L
        self.c_H = c_H
        self.c_hidden = c_hidden

        self.ce = CEBlock(self.c_H)

        self.left1 = nn.Sequential(
            nn.Conv2d(
                self.c_L, self.c_L, kernel_size=3, stride=1,
                padding=1, groups=self.c_L, bias=False),
            nn.BatchNorm2d(self.c_L),
            nn.Conv2d(
                self.c_L, self.c_L, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                self.c_L, self.c_L, kernel_size=3, stride=2,
                padding=1, bias=False),
            # nn.BatchNorm2d(self.c_L), # no bn in tf version
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                self.c_H, self.c_L, kernel_size=3, stride=1,
                padding=1, bias=False),
            # nn.BatchNorm2d(self.c_L), # same as line 58
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                self.c_H, self.c_H, kernel_size=3, stride=1,
                padding=1, groups=self.c_H, bias=False),
            nn.BatchNorm2d(self.c_H),
            nn.Conv2d(
                self.c_H, self.c_L, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                self.c_L, self.c_L, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(self.c_L),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, X):
        x_d = X[0] # detailed branch (fL)
        x_s = X[1] # semantic branch (fH)

        x_s, x_gap = self.ce(x_s)
        x_gap = x_gap.view(-1, self.c_H)

        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out
