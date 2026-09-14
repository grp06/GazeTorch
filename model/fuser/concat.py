import torch
import torch.nn as nn
from model.module_base import Module_base
from functools import reduce


class Simple_concat_fuser(Module_base):
    """
    channels: list of #Cs. #C_in => #C_hidden1 => ... => #C_out
    """

    def __init__(self, name, input_shape, **kwargs):
        Module_base.__init__(self, name=name, input_shape=input_shape)

    def forward(self, X):
        # flatten each single inputs
        X = [torch.flatten(x, start_dim=1, end_dim=-1) for x in X]
        return torch.concat(X, 1)

class Map_vector_stacker(Module_base):
    """
    MVS takes some 2D maps and 1D vectors as input.
    1D vectors are all tiled to fit map size, then stacked together.
    e.g. X = [Tensor(N,3,5,5), Tensor(N,4)] => Y = [Tensor(N,3+4,5,5)]
    """

    def __init__(self, name, input_shape, **kwargs):
        Module_base.__init__(self, name=name, input_shape=input_shape)

    def forward(self, X):
        # flatten each single inputs
        for x in X:
            if x.ndim == 4:
                n,c,h,w = x.shape
            break
        for i, x in enumerate(X):
            if x.ndim < 4:
                for _ in range(4-x.ndim):
                    x = x[..., None]
                tile_arg = [1,1,h//x.shape[2],w//x.shape[3]]
                x = x.tile(tile_arg)
                X[i] = x
        return torch.concat(X, 1)
