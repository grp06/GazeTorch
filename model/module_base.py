import torch.nn as nn
import numpy as np
import torch
import sys
sys.path.append("..")


class Module_base(nn.Module):
    """
    Usage of Module_base class:
    
    __init__
    name: module name identifier (registry at module_loader.py)
    input_shape: list[list[int]], specifying the shape of each input tensor
    --Example--
    for a backbone taking an 60x36x3 image img,
        input_shape should be [[3, 60, 36]]

    forward(X)
    X should be a LIST of input tensor (tensors in a specific order)
    in the example above, X=[img]
    """

    def __init__(self, name: str, input_shape) -> None:
        nn.Module.__init__(self)
        self.input_shape = input_shape
        self.name = name

    def get_output_shape(self):
        # TODO: verify whether device="cuda" will malfunction when using data parallel
        dummy_inputs = [torch.randn(*s, device="cuda") for s in self.input_shape]
        dummy_output = self.forward(dummy_inputs)
        if isinstance(dummy_output, torch.Tensor):
            return [list(dummy_output.size())]
        return [list(dummy.size()) for dummy in dummy_output]

    def forward(self, X):
        # abstract method, should never be called
        raise Exception("forward method not implemented!!")

    def count_flops(self) -> int:
        # generate dummy inputs according to the given input shape during initializating to count the flops
        from fvcore.nn.flop_count import flop_count
        dummy_inputs = [torch.randn(*s, device="cuda") for s in self.input_shape]
        counter, *_ = flop_count(self, dummy_inputs)
        return sum(counter.values())

    def count_params(self, ignore_bn=False) -> int:
        if not ignore_bn:
            return np.sum([p.numel() for p in self.parameters()]).item()
        else:
            count = 0
            for m in self.modules():
                if not isinstance(m, nn.BatchNorm3d):
                    for p in m.parameters(recurse=False):
                        count += p.numel()
        return count
