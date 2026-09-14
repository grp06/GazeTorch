import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss


class Symmetry_loss(_Loss):
    '''
    for n_views=2 + flip_every_other_view case only.
    pitch: if True, then impose pitch consistency loss (view #1's pitch == view #2's)
    '''
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean',
                 pitch=False, yaw=True):
        super().__init__(size_average, reduce, reduction)
        self.pitch = pitch
        self.yaw = yaw
        self.L1 = nn.L1Loss()
        
    def forward(self, x:Tensor):
        # x: [B, 2] gaze tensor
        b = x.shape[0]
        view1 = x[:b//2, :]
        view2 = x[b//2:, :]

        loss = 0

        if self.yaw:
            yaw_gt = -view2[:,1]
            yaw_gt = yaw_gt.detach()
            loss += self.L1(view1[:,1], yaw_gt)

        if self.pitch:
            pitch_gt = view2[:,0]
            pitch_gt = pitch_gt.detach()
            loss += self.L1(view1[:,0], pitch_gt)
        
        return loss