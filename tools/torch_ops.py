import torch
import torch.nn as nn
import numpy as np


def pitchyaw_to_xyz(pitchyaw):
    '''
    :param pitchyaw: Nx2 (pitch-yaw angles in radians)
    :return: Nx3 (x-y-z)
    '''
    pitch = pitchyaw[..., 0]
    yaw = pitchyaw[..., 1]
    return torch.stack((torch.cos(pitch) * torch.sin(yaw), torch.sin(pitch), torch.cos(yaw) * torch.cos(pitch))).t()

def xyz_to_pitchyaw(xyz):
    '''
    :param xyz: Nx3 (x-y-z)
    :return: Nx2 (pitch-yaw angles in radians)
    '''
    n = torch.norm(xyz, dim=1, keepdim=True)
    xyz = xyz/n
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    return torch.stack((torch.asin(y), torch.atan(x / z))).t()

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


def angle_transform(input: torch.Tensor, n_bins: int, bin_size: int, pitch_bias_transform: float = 0.0,
                    temperature = 1., var=False):
    '''
    Convert a model's raw bin-vector output to radian, along a single axis.
    * if var=False (default), output the radian.
    * if var=True, output variance loss value.
    '''
    import torch.nn.functional as F
    batch_num = input.shape[0]
    idx_bins = np.array([idx for idx in range(n_bins)], dtype=np.float32)
    idx_bins = torch.from_numpy(idx_bins)
    idx_bins = torch.unsqueeze(idx_bins, 0)
    # idx_bins = torch.tile(idx_bins, (batch_num, 1)).to(device='cuda') # legacy
    idx_bins.repeat(batch_num, 1) # torch.tile not supported in onnx converter
    
    device_run = input.device # check device usage for: model conversion / alex eval

    if temperature != 1:
        input /= temperature

    idx_bins = idx_bins.to(device_run)
    prob = F.softmax(input, dim=1).to(device_run)
    expect_raw = torch.sum(prob * idx_bins.to(device_run), dim=1, keepdim=True)
    # print("input shape:", input.shape)
    # print("prob.shape:", prob.shape)
    # print("expect_raw.shape:", expect_raw.shape)

    if var:
        var_vector = torch.sum(prob * (idx_bins - expect_raw) ** 2, dim=1, keepdim=True)
        return torch.mean(var_vector)

    input_expect = expect_raw*bin_size - \
        torch.ones(batch_num, 1, dtype=torch.float32).to(device_run)*(n_bins // 2 * bin_size) + \
            pitch_bias_transform
    input_expect = input_expect * np.pi / 180.

    return input_expect


def count_params(model:nn.Module, ignore_bn=False) -> int:
    if not ignore_bn:
        return np.sum([p.numel() for p in model.parameters()]).item()
    else:
        count = 0
        for m in model.modules():
            if not isinstance(m, nn.BatchNorm3d):
                for p in m.parameters(recurse=False):
                    count += p.numel()
    return count