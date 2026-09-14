import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
import tools.torch_ops as ops
import torch.nn.functional as F


class L1_loss(_Loss):
    '''
    L1 error
    '''
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(L1_loss, self).__init__(size_average, reduce, reduction)
        self.reduction = reduction
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        if len(target) == 2: # here gt should be ['gaze', 'dataset_id']
            dataset_id = target[-1]
            dataset_mask = (dataset_id == 0).to(int) # xgaze -> 1, gc3d, gazegene -> 0
            input[:, 0] = input[:, 0] * dataset_mask # masked on pitch value
            target[0][:, 0] = target[0][:, 0] * dataset_mask

            loss = F.l1_loss(input, target[0], reduction=self.reduction)
        else:
            loss = F.l1_loss(input, target, reduction=self.reduction)

        return loss
