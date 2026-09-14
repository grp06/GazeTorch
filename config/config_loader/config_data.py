from config.config_loader.config_base import Config_base
from tools.param_constant import *


class Config_data(Config_base):
    def __init__(self, raw_cfg):
        super().__init__(raw_cfg, DATA)
        self.__dict__[DATASET_TRAIN] = Config_data_dataset(self.read_mandatory(DATASET_TRAIN), DATASET_TRAIN)
        if DATASET_VAL in raw_cfg:
            self.__dict__[DATASET_VAL] = Config_data_dataset(self.read_mandatory(DATASET_VAL), DATASET_VAL)
            self.clean_up()
        else:
            self.__dict__[DATASET_VAL] = None

class Config_data_dataset(Config_base):
    def __init__(self, raw_cfg, key):
        super().__init__(raw_cfg, key)
        self.__dict__[TYPE] = self.read_mandatory(TYPE)
        self.__dict__[ARGS] = Config_data_dataset_args(self.read_mandatory(ARGS))

        self.clean_up()

class Config_data_dataset_args(Config_base):
    def __init__(self, raw_cfg, key =  ARGS):
        super().__init__(raw_cfg, key)
        # for k, v in list(raw_cfg.items()):
        #     if isinstance(v, dict):
        #         self.__dict__[k] = Config_data_dataset_args(v, k)
        #         del raw_cfg[k]
    
        self.clean_up()


