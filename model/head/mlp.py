import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module_base import Module_base
from tools.torch_ops import angle_transform

class MLP(Module_base):
    """
    channels: list of #Cs. #C_in => #C_hidden1 => ... => #C_out

    mode: str
    * '': default
    * 'group2': split X into pitch/yaw. perform respective MLP.
        * requires nbins_pitch, nbins_yaw; finally, returns concat of pitch/yaw features.
            * len = nbins_pitch + nbins_yaw
    * 'pitch_only/yaw_only': perform MLP on the first/latter half only.
    """

    def __init__(self, name, input_shape, channels, activation_fn=F.leaky_relu, drop_out: float = 0.0, hidden_out: bool=False,
                 mode='', nbins_pitch=9, nbins_yaw=13,
                **kwargs):
        self._validate_init(input_shape, channels)
        Module_base.__init__(self, name=name, input_shape=input_shape)
        self.activation_fn = activation_fn
        self.drop_out = drop_out
        # Get the shape of the input tensor
        input_dim = input_shape[0][0]
        self.n_layers = len(channels)
        self.hidden_out = hidden_out
        if 'pkl_args' in kwargs:
            self.pkl_args = kwargs['pkl_args']

        self.normalize = kwargs.get("normalize", False)
        
        # Create FC layers
        self.mode = mode
        if self.mode == 'group2':
            for i, c in enumerate(channels, start=1):
                if i==self.n_layers: break
                self.add_module(f"fc_p_{i}", nn.Linear(input_dim//2, c//2))
                self.add_module(f"fc_y_{i}", nn.Linear(input_dim//2, c//2))
                input_dim = c
            self.add_module(f"fc_p_{self.n_layers}", nn.Linear(input_dim//2, nbins_pitch))
            self.add_module(f"fc_y_{self.n_layers}", nn.Linear(input_dim//2, nbins_yaw))
        else:
            if 'only' in self.mode:
                input_dim = input_dim // 2
            for i, c in enumerate(channels, start=1):
                layer = nn.Linear(input_dim, c)
                self.add_module("fc%d" % i, layer)
                input_dim = c

    def _validate_init(self, input_shape, channels):
        if len(channels) == 0:
            raise Exception("[Head_mlp]:Invalid channel #s")
        if len(input_shape) != 1:
            raise Exception("[Head_mlp]:Cannot accept more than one input")
        if len(input_shape[0]) != 1:
            raise Exception("[Head_mlp]:Input has to be flattened")

    def forward(self, X):
        """
        Following Module_base, X: list[torch.Tensor]
        """
        if not hasattr(self, 'mode'):  # compatibility with old PKLs
            self.mode = ''
        if self.mode == 'group2':
            X = self.forward_split(X)
        else:
            X = X[0]
            if getattr(self, 'normalize', False):
                X = torch.nn.functional.normalize(X, p=2.0, dim=1)
            if self.mode == 'pitch_only':
                n_channels = X.shape[1]
                X = X[:, :n_channels//2]
            elif self.mode == 'yaw_only':
                n_channels = X.shape[1]
                X = X[:, n_channels//2:]
            # actual forward propagation
            for i in range(1, self.n_layers):
                layer = getattr(self, "fc%d" % i)
                X = layer(X)
                X = self.activation_fn(X)
                if self.drop_out != 0:
                    X = F.dropout(X, self.drop_out)
            layer = getattr(self, "fc%d" % self.n_layers)
            if getattr(self, "hidden_out", False): # use getattr() to add more features,otherwise loading .pkl model will fail.
                X_hidden = X
                X = layer(X_hidden)
                return [X, X_hidden]
            
            X = layer(X)
        
        if getattr(self, "pkl_args", False):
            temperature = self.pkl_args.get('temperature', 1.)
            max_out = self.pkl_args.get('max_out', False)
            if max_out:
                pitch_bin = X[:, :self.pkl_args['nbins_pitch']]
                yaw_bin = X[:, self.pkl_args['nbins_pitch']:self.pkl_args['nbins_yaw']+self.pkl_args['nbins_pitch']]

                pitch_max = pitch_bin * (pitch_bin == pitch_bin.amax(dim=1, keepdim=True))
                yaw_max = yaw_bin * (yaw_bin == yaw_bin.amax(dim=1, keepdim=True))

                pitch_angle = angle_transform(pitch_max, self.pkl_args['nbins_pitch'], self.pkl_args['bin_size'], self.pkl_args['pitch_bias'], temperature)
                yaw_angle = angle_transform(yaw_max, self.pkl_args['nbins_yaw'], self.pkl_args['bin_size'], temperature)

                angle_output_max_bin = torch.cat([pitch_angle, yaw_angle], dim=1)
                return angle_output_max_bin
            else:
                pitch_angle = angle_transform(X[:, :self.pkl_args['nbins_pitch']], self.pkl_args['nbins_pitch'], self.pkl_args['bin_size'], self.pkl_args['pitch_bias'], temperature)
                yaw_angle = angle_transform(X[:, self.pkl_args['nbins_pitch']:self.pkl_args['nbins_yaw']+self.pkl_args['nbins_pitch']], self.pkl_args['nbins_yaw'], self.pkl_args['bin_size'], temperature)
                angle_output = torch.cat([pitch_angle, yaw_angle], dim=1)
                return angle_output
        return X

    def forward_split(self, X):
        X = X[0]
        n_channels = X.shape[1]

        X1 = X[:, :n_channels//2]
        X2 = X[:, n_channels//2:]

        for i in range(1, self.n_layers):
            layer = getattr(self, f"fc_p_{i}")
            X1 = layer(X1)
            X1 = self.activation_fn(X1)
            if self.drop_out != 0:
                X1 = F.dropout(X1, self.drop_out)
        layer = getattr(self, f"fc_p_{self.n_layers}")
        X1 = layer(X1)

        for i in range(1, self.n_layers):
            layer = getattr(self, f"fc_y_{i}")
            X2 = layer(X2)
            X2 = self.activation_fn(X2)
            if self.drop_out != 0:
                X2 = F.dropout(X2, self.drop_out)
        layer = getattr(self, f"fc_y_{self.n_layers}")
        X2 = layer(X2)

        X = torch.cat([X1, X2], dim=1)
        return X