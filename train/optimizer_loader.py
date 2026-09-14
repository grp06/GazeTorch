import torch

optimizers = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

lr_schedulers = {
    "const_decay": torch.optim.lr_scheduler.ConstantLR,
    "step": torch.optim.lr_scheduler.StepLR,
    "exp": torch.optim.lr_scheduler.ExponentialLR
}

def get_optimizer(key):
    return optimizers[key]

def get_lr_scheduler(key="None"):
    return lr_schedulers[key]