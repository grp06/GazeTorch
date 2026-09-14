import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module_base import Module_base
from tools.torch_ops import angle_transform

class Regressor(Module_base):
    """
    - input: (B, nbins_pitch + nbins_yaw)
    - output: (B, 2)
    """

    def __init__(self, name:str, 
                input_shape,
                nbins_pitch:int = 9, 
                nbins_yaw:int = 13, 
                bin_size:int = 4, 
                beta_y:float = 10.0,
                beta_p:float = 10.0,
                pitch_bias:float = 0.0,
                temperature:float = 0.5):
        
        # define class attributes
        self.nbins_pitch = nbins_pitch
        self.nbins_yaw = nbins_yaw
        self.temperature = temperature
        self.bin_size = bin_size
        self.pitch_bias = pitch_bias

        self._validate_init(input_shape)
        Module_base.__init__(self, name=name, input_shape=input_shape)

    def _validate_init(self, input_shape):
        if input_shape[0][0] != self.nbins_pitch + self.nbins_yaw:
            raise Exception("[Regressor]:Input does not match (nbins_pitch + nbins_yaw) dimensions!")

    def forward(self, X) -> list[torch.Tensor]:
        """
        Following Module_base, X: list[torch.Tensor]
        """
        X = X[0]

        input_pitch = X[:, :self.nbins_pitch] # [N, nbins_pitch]
        input_yaw = X[:, self.nbins_pitch:self.nbins_pitch + self.nbins_yaw] # [N, nbins_yaw]

        pitch_angle = angle_transform(input_pitch, self.nbins_pitch, self.bin_size, self.pitch_bias, self.temperature) # [N, 1]
        yaw_angle = angle_transform(input_yaw, self.nbins_yaw, self.bin_size, self.temperature) # [N, 1]

        return torch.cat([pitch_angle, yaw_angle], dim=1)
        