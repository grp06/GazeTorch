import torch
import numpy as np

from torch import Tensor
from torch.nn.modules.loss import _Loss
import tools.torch_ops as ops

class Angular_loss(_Loss):
    '''
    Angular error (via cosine similarity).
    '''
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(Angular_loss, self).__init__(size_average, reduce, reduction)
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        '''
        :param input: Nx2 tensor.
        :param target: Nx2 tensor.
        :return: angular error (in degrees).
        '''
        a = ops.pitchyaw_to_xyz(input)
        b = ops.pitchyaw_to_xyz(target)
        ab = torch.sum(a*b, dim=-1)
        ab = torch.clamp(ab, max=1.)

        d = torch.acos(ab) * 180.0 / np.pi
        if self.reduction == 'mean':
            d = torch.mean(d)
        elif self.reduction == 'sum':
            d = torch.sum(d)
        return d