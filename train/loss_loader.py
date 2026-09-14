import torch
from tools.printer import Color_string
import importlib
# from train.loss.CR import CR_loss
# from train.loss.RSD import RSD_loss


_private_imports = {
    "l1": "train.loss.l1.L1_loss",
    # "l1": "torch.nn.L1Loss",
    "l2": "torch.nn.MSELoss",
    "angular": "train.loss.angular.Angular_loss",
    "bce": "torch.nn.BCELoss",
    "dice": "train.loss.dice.DiceLoss",
    "simsiam": "train.loss.simsiam.SimSiam_loss",
    "supcon": "train.loss.supcon.SupConLoss",
    "binsupcon": "train.loss.supcon.Bin_SupConLoss",
    "binsupcon_v2": "train.loss.supcon.Bin_SupConLoss_v2",
    "car": "train.loss.car.Car_loss",
    "symmetry": "train.loss.symmetry.Symmetry_loss",
}

def get_loss(loss_name, **kwargs):
    if loss_name in _private_imports:
        loss_path, loss_class = _private_imports[loss_name].rsplit(".", 1)
        loss_module = importlib.import_module(loss_path)
        loss = getattr(loss_module, loss_class)
        return loss(**kwargs)
    else:
        raise Exception(Color_string.fail("[loss]Loss function not found:%s" % loss_name))