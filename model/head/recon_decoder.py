import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module_base import Module_base


class resnet_decoder_all(Module_base):
    def __init__(self, name, input_shape, **kwargs):
        # self._validate_init(input_shape)
        Module_base.__init__(self, name=name, input_shape=input_shape)

        self.layer4_1x1 = self.convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = self.convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = self.convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = self.convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = self.convrelu(64 + 256, 128, 3, 1)

        self.conv_last1 = self.convrelu(128, 64, 3, 1)

        self.conv_last2 = nn.Conv2d(64, 3, 1)

    def _validate_init(self, input_shape):
        if len(input_shape) != 1:
            raise Exception("[Head_recon_decoder]:Cannot accept more than one input")
        if len(input_shape[0]) != 1:
            raise Exception("[Head_recon_decoder]:Input has to be flattened")
        
    def convrelu(self, in_channels, out_channels, kernel, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        '''
            Input is a list of feature maps from Resnet18, has 5 elements:
                ['output from conv1', 'output from relu', 'layer1', 'layer2', 'layer3'],
            where layer1,2,3 are from official Resnet18 defination(from '_make_layer' method).

        '''

        layer0 = input[2]
        layer1 = input[3]
        layer2 = input[4]
        layer3 = input[5]
        layer4 = input[6]

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = self.conv_last1(x)
        out = self.conv_last2(x)

        return out