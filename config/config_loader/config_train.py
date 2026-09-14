from collections import OrderedDict
from tools.cfgnode import CfgNode
from config.config_loader.config_base import Config_base
from tools.param_constant import *


class Config_train(Config_base):
    def __init__(self, raw_cfg, key=TRAIN):
        super().__init__(raw_cfg, key)
        self.__dict__[LOSS] = Config_train_loss(self.read_mandatory(LOSS))
        self.__dict__[OPTIMIZER] = Config_train_optimizer(self.read_mandatory(OPTIMIZER))
        for k, v in list(raw_cfg.items()):
            if isinstance(v, dict):
                self.__dict__[k] = Config_train(v, k)
                del raw_cfg[k]
        self.clean_up()


class Config_train_optimizer(Config_base):
    def __init__(self, raw_cfg, key=OPTIMIZER):
        super().__init__(raw_cfg, key)
        self.__dict__[DEFAULT]: Config_train_optimizer_blackbox(self.read_mandatory(DEFAULT), DEFAULT)
        for k, v in list(raw_cfg.items()):
            if isinstance(v, dict):
                self.__dict__[k] = Config_train_optimizer_blackbox(v, k)
                del raw_cfg[k]
        self.clean_up()

class Config_train_optimizer_blackbox(Config_base):
    def __init__(self, raw_cfg, name):
        super().__init__(raw_cfg, name)
        self.__dict__[TYPE] = self.read_mandatory(TYPE)
        self.__dict__[LR] = self.read_mandatory(LR)
        # since we keep it as orderedDict, recursive reading is essentially, not available 
        # (at least heavily discouraged)
        args = self.read_mandatory(ARGS)
        if args is None:
            self.__dict__[ARGS] = None 
        else:
            self.__dict__[ARGS] = OrderedDict(args) 

        lrs = self.read_non_mandatory(LRS)
        if lrs is not None:
            self.__dict__[LRS] = CfgNode(lrs)

        self.clean_up()



class Config_train_loss(Config_base):
    def __init__(self, raw_cfg, key=LOSS):
        super().__init__(raw_cfg, key)
        self.__dict__[LOSS_TRAIN] = OrderedDict(self.read_mandatory(LOSS_TRAIN))
        validation_loss = self.read_non_mandatory(LOSS_VAL)
        if validation_loss is not None:
            self.__dict__[LOSS_VAL] = OrderedDict(validation_loss) 
        for k, v in list(raw_cfg.items()):  # assuming rest of are different named losses
            if isinstance(v, dict):
                self.__dict__[k] = Config_train_loss_blackbox(v, k)
                del raw_cfg[k]
        self.clean_up()


class Config_train_loss_blackbox(Config_base):
    def __init__(self, raw_cfg, name):
        super().__init__(raw_cfg, name)
        self.__dict__[MODULE_INPUT] = OrderedDict(self.read_mandatory(MODULE_INPUT))
        self.__dict__[TYPE] = self.read_mandatory(TYPE)
        self.clean_up()
