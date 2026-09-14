import torch.nn as nn
from model.module_base import Module_base
import torchvision.models.mobilenet as Net


class Mobilenet_V2(Module_base):
    def __init__(self, name, input_shape, **kwargs) -> None:
        # ideally validate parameter (reference MLP)
        Module_base.__init__(self, name=name, input_shape=input_shape)
        self.net = Net.mobilenet_v2(**kwargs)

    def forward(self, X): # X: list[Tensor]
        # ideally introduce validation for size here but because the entire class might get changed I will not add it here (reference MLP)
        X = X[0]
        return self.net(X)
