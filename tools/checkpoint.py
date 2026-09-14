'''
[Protocol]
A ckpt contains:
    model (model): model to save the weight to the checkpoint.
    optimizer (optim): optimizer to save the historical state.
    epoch (int): current number of epoch of the model.
    cfg (CfgNode): configs to save.
In a ckpt_dir, each ckpt is named like "ckpt_epoch_<id= :05d>.ckpt".
'''

from ast import Raise
import os
from typing import OrderedDict
import torch
from glob import glob

from tools.distributed import Distributed_utils as du
from tools.printer import Printer

CKPT_NAME = "ckpt_epoch_{:05d}.pyth"
CKPT_KEYWORD = ".pyth"
TB_KEYWORD = "events.out*"

class Checkpoint_io:
    def __init__(self, ckpt_dir, parallel=True):
        self.ckpt_dir = ckpt_dir
        self.parallel = parallel

    def get_last_ckpt(self, ckpt_dir):
        ckpts = os.listdir(ckpt_dir) if os.path.exists(ckpt_dir) else []
        ckpts = [fn for fn in ckpts if CKPT_KEYWORD in fn]
        if len(ckpts)==0:
            Printer.print_WARNING("[Checkpoint]Invalid dir/No ckpt found in the dir.")
            return None
        fn = sorted(ckpts)[-1]
        return os.path.join(ckpt_dir, fn)

    def has_ckpt(self):
        if self.ckpt_dir is None: return False
        ckpts = os.listdir(self.ckpt_dir) if os.path.exists(self.ckpt_dir) else []
        return any(CKPT_KEYWORD in fn for fn in ckpts)

    # disabled optimizer saving temporarily
    def save_ckpt(self, model, epoch, cfg=None):
        if self.parallel and not du.is_master_proc():
            return
        if self.ckpt_dir is None:
            err_str = "[checkpoint]: ckpt_dir cannot be None"
            Printer.print_FAIL(err_str)
            Raise(err_str)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        state_dict = model.module.state_dict() if self.parallel else model.state_dict()
        checkpoint = {
            "epoch": epoch,
            "model_state": state_dict,
            # "optimizer_state": optimizer.state_dict(),
            "cfg": cfg.dump() if cfg else None, # backup cfg. To do: current cfg has not dump() method.
        }
        ckpt_path = os.path.join(self.ckpt_dir, CKPT_NAME.format(epoch))
        torch.save(checkpoint, ckpt_path)
        return ckpt_path


    def load_ckpt(self, ckpt_name, model, overriding_ckpts=None, optimizer=None, parallel=False):
        '''
        :param ckpt_name: ckpt path (relative). If ckpt_name=None then automatically use the last ckpt.
        :param model: model<-ckpt
        :param parallel: if in parallel mode
        :param optimizer: optimizer<-ckpt
        :return:
        '''
        m = model.module if parallel else model
        checkpoint = self.get_ckpt_state(self.ckpt_dir, ckpt_name) if self.ckpt_dir is not None else self.get_default_ckpt(m)

        if overriding_ckpts:
            for name, ckpt in overriding_ckpts.items():
                overriding_ckpt = self.get_ckpt_state(ckpt.ckpt_module_dir, ckpt.ckpt_module_name)["model_state"]
                for k, v in overriding_ckpt.items():
                    if k.split('.')[1] == name : # check if this is part of the checkpoint 
                        if k not in checkpoint["model_state"]:
                            err_str = "overriding module failed, key word {} does not exists in the current module".format(k)
                            Printer.print_FAIL(err_str)
                            Raise(err_str)
                        checkpoint["model_state"][k] = v
                Printer.print_WARNING("overrided module {}".format(name))
        return self.load_module(m, checkpoint, optimizer)

    def get_default_ckpt(self, model):
        return {
            "epoch": 0,
            "model_state": model.state_dict(),
            # "optimizer_state": optimizer.state_dict(),
            # "cfg": cfg.dump() if cfg else None, # backup cfg. To do: current cfg has not dump() method.
        }

    def get_ckpt_state(self, ckpt_dir, ckpt_name=None):
        ckpt_path = os.path.join(ckpt_dir, ckpt_name) if ckpt_name else self.get_last_ckpt(ckpt_dir)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("[Checkpoint]Invalid ckpt path: {}".format(ckpt_path))
        return torch.load(ckpt_path, map_location="cpu")

    def load_module(self, model, checkpoint, optimizer=None):
        model.load_state_dict(checkpoint["model_state"], strict=False) # Amirul's bugfix - for fine-tuning
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        epoch = checkpoint["epoch"]
        return epoch
        
    def _clear(self, keyword, msg):
        fns = glob(os.path.join(self.ckpt_dir, keyword))
        if du.is_master_proc() and len(fns) > 0:
            for fn in fns:
                os.remove(fn)
            Printer.print_OK("[checkpoint]{} removed.".format(msg))

    def clear_dir(self, clear_ckpts=True, clear_tb=True):
        if clear_ckpts:
            self._clear(CKPT_KEYWORD, "Checkpoints")
        if clear_tb:
            self._clear(TB_KEYWORD, "TB logs")