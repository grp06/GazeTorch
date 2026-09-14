import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module_base import Module_base
from functools import reduce

from tools.torch_ops import ConvBNReLU

class SEG(Module_base):
    def __init__(self, name, input_shape, eyelid=False, **kwargs):
        self._validate_init(input_shape)
        Module_base.__init__(self, name=name, input_shape=input_shape)

        n_iris_classes = 2
        n_eye_classes = 3 if eyelid else 2

        # seg heads with Amirul's class
        self.head_eye_L = SegmentHeadEye(input_shape[0][0], 1024, n_eye_classes, up_factor=8, aux=False)
        self.head_iris_L = SegmentHeadIris(input_shape[0][0], 1024, n_iris_classes, up_factor=8, aux=False)
        self.head_eye_R = SegmentHeadEye(input_shape[0][0], 1024, n_eye_classes, up_factor=8, aux=False)
        self.head_iris_R = SegmentHeadIris(input_shape[0][0], 1024, n_iris_classes, up_factor=8, aux=False)

    def _validate_init(self, input_shape):
        if len(input_shape) != 1:
            raise Exception("[Head_seg]:Cannot accept more than one input")
        if len(input_shape[0]) != 1:
            raise Exception("[Head_seg]:Input has to be a 3D Tensor. Got ", input_shape[0])

    def forward(self, X):
        """
        Following Module_base, X: list[torch.Tensor]

        """
        X = X[0]

        # actual forward propagation 
        logits_eye_L = self.head_eye_L(X)
        logits_iris_L = self.head_iris_L(X)
        logits_eye_R = self.head_eye_R(X)
        logits_iris_R = self.head_iris_R(X)

        seg_maps = torch.cat([logits_eye_L, logits_iris_L, logits_eye_R, logits_iris_R], dim=1)
        
        return seg_maps


class SegmentHeadEye(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHeadEye, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
                ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat

class SegmentHeadIris(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHeadIris, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
                ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat
