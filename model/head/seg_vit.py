'''
A wrapper for Tensorboard ViTs. Add "neck" from:
https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
L88.
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from model.module_base import Module_base
from model.head.sam.mask_decoder import MaskDecoder
from model.head.sam.transformer import TwoWayTransformer
from model.head.sam.prompt_encoder import PositionEmbeddingRandom


ORIGINAL_SIZE = (224,224)

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    

class ViT_Seg(Module_base):
    def __init__(self, name, input_shape, **kwargs) -> None:
        # ideally validate parameter (reference MLP)
        Module_base.__init__(self, name=name, input_shape=input_shape)

        embed_dim = kwargs.get("channels", 768) # B = 768, L = 1024, H = 1280
        out_chans = kwargs.get("hidden_channels", 256)
        patch_size = kwargs.get("patch_size", 14) # for B/16, shoule be 224/16=14. (w in forward() should also be equal to this)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.dec = MaskDecoder(
            transformer_dim=out_chans,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=out_chans,
                mlp_dim=2048,
                num_heads=8,
            )
        )

        self.pe = PositionEmbeddingRandom(out_chans // 2)
        self.image_pe = self.pe((patch_size, patch_size)).unsqueeze(0)

    def forward(self, X): # X = [X_seg]
        X_seg = X[0]
        b, wh, d = X_seg.shape
        w = int(math.sqrt(wh))

        X_seg = X_seg.view(b, d, w, w).contiguous()
        # pos_emb = pos_emb.view(-1, d, w, w).contiguous()

        x = self.neck(X_seg) # (b,d,w,w), "image_embeddings"

        print("PE:", self.image_pe.shape)

        masks = self.dec(x, self.image_pe) # (b,4,4w,4w)

        # upsample (postprocess_masks)
        masks = F.interpolate(masks, ORIGINAL_SIZE, mode='bilinear', align_corners=False)

        print("mask's shape:", masks.shape)

        return masks


if __name__=='__main__':
    M = ViT_Seg('head', [])
    
    X_seg = torch.zeros(2, 196, 768)
    pos_emb = torch.zeros(1, 196, 768)

    X = [X_seg, pos_emb]
    y = M(X)