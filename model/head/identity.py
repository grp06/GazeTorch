import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module_base import Module_base


class Identity(Module_base):
    def __init__(self, name, input_shape, **kwargs):
        Module_base.__init__(self, name=name, input_shape=input_shape)
        
    def forward(self, X):
        return X