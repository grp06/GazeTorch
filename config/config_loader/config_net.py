from collections import OrderedDict
from config.config_loader.config_base import Config_base
from tools.param_constant import *


class Config_net(Config_base):
    def __init__(self, raw_cfg):
        super().__init__(raw_cfg, NET)

        self.__dict__[OUTPUT] = self.read_mandatory(OUTPUT)
        self.__dict__[INPUT] = self.read_mandatory(INPUT)
        #self.__dict__[CKPT] = Config_net_ckpt(self.read_mandatory(CKPT))
        ckpt = self.read_non_mandatory(CKPT)
        if ckpt:
            self.__dict__[CKPT] = Config_net_ckpt(ckpt)
        # remaining are individual components

        # note the rest should just be all the blackboxes, such as backbone, fuser, etc
        for k, v in list(raw_cfg.items()):
            self.__dict__[k] = Config_net_blackbox(v, k)
            del raw_cfg[k]
        self.clean_up()


class Config_net_ckpt(Config_base):
    def __init__(self, raw_cfg):
        super().__init__(raw_cfg, CKPT)
        self.__dict__[DIR] = self.read_mandatory(DIR)
        self.__dict__[NAME] = self.read_non_mandatory(NAME)

        self.clean_up()

    def set_net_ckpt(self, ckpt_dir, ckpt_name):
        self.__dict__[DIR] = ckpt_dir
        self.__dict__[NAME] = ckpt_name

class Config_net_blackbox(Config_base):
    def __init__(self, raw_cfg, key):
        super().__init__(raw_cfg, key)

        self.__dict__[MODULE_INPUT] = Config_net_blackbox_module_input(self.read_mandatory(MODULE_INPUT))
        self.__dict__[MODULE_OUTPUT] = self.read_mandatory(MODULE_OUTPUT)
        self.__dict__[TYPE] = self.read_mandatory(TYPE)
        args = self.read_non_mandatory(ARGS)
        if args:
            self.__dict__[ARGS] = Config_net_blackbox_args(args)
        if CKPT_MODULE in raw_cfg:
            self.__dict__[CKPT_MODULE] = Config_net_blackbox_ckpt(self.read_mandatory(CKPT_MODULE))

        self.clean_up()


class Config_net_blackbox_ckpt(Config_base):
    def __init__(self, raw_cfg):
        super().__init__(raw_cfg, CKPT)
        self.__dict__[CKPT_MODULE_DIR] = self.read_mandatory(CKPT_MODULE_DIR)
        self.__dict__[CKPT_MODULE_NAME] = self.read_non_mandatory(CKPT_MODULE_NAME)
        self.clean_up()


class Config_net_blackbox_module_input(Config_base):
    def __init__(self, raw_cfg):
        super().__init__(raw_cfg, INPUT)
        self.clean_up()


class Config_net_blackbox_args(Config_base):
    def __init__(self, raw_cfg, key=ARGS):
        super().__init__(raw_cfg, key)
        for k, v in list(raw_cfg.items()):
            if isinstance(v, dict):
                self.__dict__[k] = Config_net_blackbox_args(v, k)
                del raw_cfg[k]
        self.clean_up()
