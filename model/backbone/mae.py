# https://github.com/facebookresearch/mae/blob/main/models_mae.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from timm.models.vision_transformer import PatchEmbed, Block
from .backbone_utils.pos_embed import get_2d_sincos_pos_embed
from model.module_base import Module_base
from tools.utils import args_to_dict


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 skip_decoder=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.skip_decoder = skip_decoder
        if not self.skip_decoder:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if not self.skip_decoder:
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        if not self.skip_decoder:
            torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


D_b16 = args_to_dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
D_l16 = args_to_dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
D_h14 = args_to_dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


# GazeTorch customized
def mae_vit_base_patch16_enc(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), skip_decoder=True, **kwargs)
    return model

def mae_vit_large_patch16_enc(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), skip_decoder=True, **kwargs)
    return model

def mae_vit_huge_patch14_enc(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), skip_decoder=True, **kwargs)
    return model

PATH_CKPT = "/data/lzh/UniGaze/ckpts/mae_b16_checkpoint-299.pth"
NUM_SEGMAPS = 8
class MAE(Module_base):
    def __init__(self, name, input_shape, **kwargs) -> None:
        # ideally validate parameter (reference MLP)
        Module_base.__init__(self, name=name, input_shape=input_shape)

        self.model_type = kwargs.get("model_type", "b16")
        self.decoder_type = kwargs.get("decoder_type", "v1")
        if self.model_type == 'b16':
            if self.decoder_type == 'v0':
                self.mae = mae_vit_base_patch16_enc()
            else:
                self.mae = mae_vit_base_patch16()
            self.mae_args = D_b16
            self.dim = 768
            self.w = 14
        elif self.model_type == 'l16':
            self.mae = mae_vit_large_patch16_enc()
            self.mae_args = D_l16
            self.dim = 1024
            self.w = 14
        elif self.model_type == 'h14': # 10.8GB VRAM
            self.mae = mae_vit_huge_patch14_enc()
            self.mae_args = D_h14
            self.dim = 1280
            self.w = 16

        # unload mae_args
        for key, value in self.mae_args.items():
            setattr(self, key, value)

        # load the ckpt
        path_ckpt = kwargs.get("pretrained", None)
        print("[MAE] model_type:", self.model_type)
        print("[MAE] path_ckpt:", path_ckpt)
        if path_ckpt is not None:
            ckpt = torch.load(path_ckpt, map_location='cpu')
            self.mae.load_state_dict(ckpt['model'], strict=False)

        # drop the loaded decoder head ... attach a new one that outputs 4 seg maps
        
        print("[MAE] Decoder_type:", self.decoder_type)
        if self.decoder_type == 'v0':
            self.skip_decoder = True
        else:
            self.build_decoder(self.decoder_type)
            self.skip_decoder = False

        self.seg = kwargs.get("seg", False)
        if self.seg:
            print("[MAE] Add Seg output.")
            self.conv_seg = nn.Linear(self.dim, 128)

    def build_decoder(self, decoder_type='v1'):
        # self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)
        # # self.decoder_pos_embed remains
        # self.decoder_blocks = nn.ModuleList([
        #     Block(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer)
        #     for i in range(self.decoder_depth)])

        # self.decoder_norm = self.norm_layer(self.decoder_embed_dim)
        if decoder_type=='v1':
            self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.patch_size**2 * NUM_SEGMAPS, bias=True) # decoder to patch
        else: # v2
            self.decoder_pred = nn.Linear(self.decoder_embed_dim, (self.patch_size // 2)**2 * NUM_SEGMAPS, bias=True) # decoder to patch
        
        for m in [self.decoder_pred]:
            m.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *NUM_SEGMAPS)
        imgs: (N, NUM_SEGMAPS, H, W)
        """
        p = self.mae.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, NUM_SEGMAPS))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], NUM_SEGMAPS, h * p, h * p))
        return imgs
    
    def unpatchify_v2(self, x):
        """
        x: (N, L, (patch_size//2)**2 *NUM_SEGMAPS)
        imgs: (N, NUM_SEGMAPS, H//2, W//2) -> interpolate -> (N, NUM_SEGMAPS, H, W)
        """
        p = self.mae.patch_embed.patch_size[0] // 2
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, NUM_SEGMAPS))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], NUM_SEGMAPS, h * p, h * p))
        imgs = F.interpolate(imgs, (h*p*2, h*p*2), mode='bilinear')
        return imgs
        
    def forward(self, X):
        x = X[0] # the image

        # ENCODE: skip random masking

        # embed patches
        x = self.mae.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.mae.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.mae.cls_token + self.mae.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.mae.blocks:
            x = blk(x)
        x = self.mae.norm(x)

        feature = x[:,0,:] # passed to gaze head

        seg = None
        if getattr(self, 'seg', False):
            seg = x[:,1:,:] # B, L, d
            seg = self.conv_seg(seg) # B, L, 128
            seg = rearrange(seg, 'd (h w) c -> d c h w', h=self.w, w=self.w)

        if getattr(self, 'skip_decoder', False):
            return feature, seg

        # DECODE: seg

        # embed tokens
        x = self.mae.decoder_embed(x)

        # add pos embed
        x = x + self.mae.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.mae.decoder_blocks:
            x = blk(x)
        x = self.mae.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        # UNPATCHIFY
        if getattr(self, 'decoder_type', 'v1') == 'v1':
            x = self.unpatchify(x) # (b,NUM_SEGMAPS,h,w)
        else:
            x = self.unpatchify_v2(x)

        seg = x

        return feature, seg
    
if __name__=='__main__':
    M = MAE('model_type', [], model_type='l16', decoder_type='v0')
    
    X = [torch.zeros(2,3,224,224)]

    feature, seg = M(X)

    print(feature.shape)
    # print(seg.shape)
