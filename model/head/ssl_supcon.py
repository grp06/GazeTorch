import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module_base import Module_base
from functools import reduce


class SSL_CONTRASTIVE(Module_base):
    def __init__(self, name, input_shape, dual_heads=True, **kwargs):
        self._validate_init(input_shape)
        Module_base.__init__(self, name=name, input_shape=input_shape)
        
        self.input_dim = input_shape[1][0]
        self.input_mid_dim = int(self.input_dim /2)
        self.out_mid_dim = int(self.input_dim /4)

        self.dual = dual_heads
        self.person_indep_module = PersonIndependentFeatures(self.input_dim, self.input_mid_dim, self.out_mid_dim)
        if self.dual:
            self.person_dep_module = PersonDependentFeatures(self.input_dim, self.input_mid_dim, self.out_mid_dim)
        # self.merge_features = MergeFeatures(self.input_dim+self.input_dim, self.input_mid_dim)

        # self.pool = pooling
        # if self.pool != 'gap':
        #     input_size = input_shape[0][1]
        #     kernel_size = input_size
        #     self.pool_layer = 

    def _validate_init(self, input_shape):
        if len(input_shape) > 2:
            raise Exception("[Head_ssl_supcon]:Cannot accept more than two input")
        if len(input_shape[0]) != 1:
            raise Exception("[Head_ssl_supcon]:Input has to be flattened")
        
    def _pool(self, feature):
        return torch.squeeze(nn.functional.adaptive_avg_pool2d(feature, 1))

    def forward(self, X):
        """
        Following Module_base, X: list[torch.Tensor]

        """
        fH = X[1]
        fL = X[0]

        # to be able to do segmentation with 8x upsample
        fH = torch.nn.functional.interpolate(fH, scale_factor=4, mode='bilinear')

        # actual forward propagation        
        fh1 = self.person_indep_module(fH)
        if self.dual:
            fh2 = self.person_dep_module(fH)
            # merge features used for segmentation
            fM = torch.concat([fh1, fh2, fL], dim=1)
        else:
            fM = torch.concat([fh1, fL], dim=1)

        fh1 = self._pool(fh1)
        if self.dual:
            fh2 = self._pool(fh2)
            return [fh1, fh2, fM]
        else:
            return [fh1, fM]


class PersonDependentFeatures(nn.Module):

    def __init__(self, in_chan, mid_chan, out_chan):
        super(PersonDependentFeatures, self).__init__()
        self.conv1 = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.conv_out = ConvBNReLU(mid_chan, out_chan, 3, stride=1)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv_out(feat)
        return feat

class PersonIndependentFeatures(nn.Module):
    def __init__(self, in_chan, mid_chan, out_chan):
        super(PersonIndependentFeatures, self).__init__()
        self.conv1 = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.conv_out = ConvBNReLU(mid_chan, out_chan, 3, stride=1)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv_out(feat)
        return feat


class MergeFeatures(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(MergeFeatures, self).__init__()
        self.conv1 = ConvBNReLU(in_chan, out_chan, 3, stride=1)
        self.conv2 = ConvBNReLU(out_chan, out_chan, 1, stride=1, padding=0)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        return feat

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class BGALayer(nn.Module):

    def __init__(self):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
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