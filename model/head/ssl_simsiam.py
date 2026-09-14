import torch.nn as nn
import torch.nn.functional as F
from model.module_base import Module_base
from functools import reduce


class SSL_SimSiam(Module_base):
    def __init__(self, name, input_shape, **kwargs):
        self._validate_init(input_shape)
        Module_base.__init__(self, name=name, input_shape=input_shape)
        
        self.projector = projection_MLP(input_shape[0][0])
        self.predictor = prediction_MLP()

    def _validate_init(self, input_shape):
        if len(input_shape) != 1:
            raise Exception("[Head_ssl]:Cannot accept more than one input")
        if len(input_shape[0]) != 1:
            raise Exception("[Head_ssl]:Input has to be flattened")

    def forward(self, X):
        """
        Following Module_base, X: list[torch.Tensor]

        """
        X = X[0]
        # actual forward propagation
        
        feat_s1 = X[:int(X.shape[0]/2)]
        feat_s2 = X[int(X.shape[0]/2):]
        # print(int(face.shape[0]/2))

        proj_s1 = self.projector(feat_s1)
        h_s1 = self.predictor(proj_s1)

        proj_s2 = self.projector(feat_s2)
        h_s2 = self.predictor(proj_s2)

        # simsiam_param = torch.cat([h_s1, h_s2, proj_s1, proj_s2], dim=1)
        
        return h_s1, h_s2, proj_s1, proj_s2



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if len(x.shape) == 4: # if x is a feature map: flatten it
            x = F.avg_pool2d(x, kernel_size=x.shape[3]).flatten(1,-1)
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 
