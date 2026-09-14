'''
A wrapper for Tensorboard ViTs. Add "neck" from:
https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
L88.

model_urls = {
    "vit_b_16": "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
    "vit_b_32": "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
    "vit_l_16": "https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
    "vit_l_32": "https://download.pytorch.org/models/vit_l_32-c7638314.pth",
}
'''
import torch

from model.module_base import Module_base


class ViT(Module_base):
    def __init__(self, name, input_shape, **kwargs) -> None:
        # ideally validate parameter (reference MLP)
        Module_base.__init__(self, name=name, input_shape=input_shape)

        self.model_type = kwargs.get("type", "b16")
        if self.model_type == 'b16':
            from torchvision.models import vit_b_16
            self.vit = vit_b_16()
            del self.vit.heads

    def forward(self, X): # X: list[Tensor]
        X = X[0]

        # get raw encoder output
        X = self.vit._process_input(X)
        b = X.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(b, -1, -1)
        X = torch.cat([batch_class_token, X], dim=1)

        X = self.vit.encoder(X) # => (b, nP^2+1, dim)

        X_cls = X[:, 0] # (b, dim)
        X_seg = X[:, 1:] # (b, nP^2, dim)
        # pos_emb = self.vit.encoder.pos_embedding # (1, nP^2+1, dim)

        # 2 outputs for gaze & seg heads
        return X_cls, X_seg