from typing import Union, List, Dict, Any, cast

import torch
import torch.nn as nn

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, in_channels=3) -> nn.Sequential:
    layers: List[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif isinstance(v, str) and v.startswith("E"):
            # the last layer of DEC. "E%d", %d = no. of channels. e.g. E1 for irismap estimation (L1).
            v = cast(int, v[1:])
            layers += [nn.Conv2d(in_channels, 1, kernel_size=9, padding=4)]
        else:
            v = cast(int, v)
            if v>0:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                # v<0 means deconv
                v = -v
                conv2d = nn.ConvTranspose2d(in_channels, v, kernel_size=4, stride=2, padding=1)
            
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG_AE(nn.Module):
    def __init__(
        self, enc: nn.Module, dec: nn.Module, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        
        # Architecture: ENC=> (feature) => Dropout => DEC => tanh [-1,1] => turn into [0,1]
        self.enc = enc
        self.dropout = nn.Dropout(p=dropout)
        self.dec = dec
        self.act = torch.tanh

        # copied from original vgg
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc(x)
        x = self.dropout(x)
        x = self.dec(x)
        x = self.act(x) + 1
        x *= 0.5
        return x

def vgg11ae(batch_norm=False, **kwargs) -> VGG_AE:
    _enc = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"] # vanilla VGG11
    _dec = [-512, -256, -128, -64, -64, "E"] # decoder

    enc = make_layers(_enc, batch_norm=batch_norm, in_channels=3)
    dec = make_layers(_dec, batch_norm=batch_norm, in_channels=512)

    return VGG_AE(enc, dec, **kwargs)