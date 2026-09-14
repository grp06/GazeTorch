import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class SimSiam_loss(nn.Module):
    '''
    Simsiam Loss (via cosine similarity).
    '''
    def __init__(self, version='simplified'):
        super(SimSiam_loss, self).__init__()
        self.version = version
        self.feature_dim = 2048

    def forward(self, p1, p2, z1, z2) :  # negative cosine similarity
        if self.version == 'simplified':# same thing as the original implementation, much faster. 
            l1 = - F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
            l2 = - F.cosine_similarity(p2, z1.detach(), dim=-1).mean()
            return l1/2 + l2/2
        else:
            raise Exception
    