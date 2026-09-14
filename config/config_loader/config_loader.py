"""
THIS IS THE ROOT FILE
you should, and you only should import config_loader.py in order to use the config_loader
This is your single, defined access point. direct import from any other support files breaks the pattern!!
"""
from tools.param_constant import *
from tools.cfgnode import CfgNode

from config.config_loader.config_net import Config_net as Net
from config.config_loader.config_net import Config_net_ckpt
from config.config_loader.config_train import Config_train as Train
from config.config_loader.config_data import Config_data as Data
from config.config_loader.config_base import Config_base

from tools.printer import Printer

import argparse
import yaml


class Config_loader(Config_base):
    """
    the initializer for the config file.
    It is only expected to have the address of config fill passed
    Config_loader will validate the basic structure of the yaml, but not
    the specific details. Most of the implementation (NET, TRAIN, DATA) details are
    not handled by config_loader

    sample usage:
        self.cfg = Config_loader(cfg_src)

    @ param
        cfg_src: a src string indicating the location of the yaml file that continas the config

    @ return
        None
    """

    def __init__(self, cfg_src):
        with open(cfg_src, "r") as f:
            Printer.print_HEADER("[config_loader]Successfully opened config file: {}".format(cfg_src))
            try:
                raw_cfg = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise exc

            Printer.print_HEADER("[config_loader]Successfully loaded config file: {} as yaml".format(cfg_src))
            self.net = Net(raw_cfg[NET])  # only pass net related content
            self.train = Train(raw_cfg[TRAIN])
            self.data = Data(raw_cfg[DATA])

    def set_net_ckpt(self, ckpt_dir, ckpt_name):
        # override cfg.net.ckpt.* with current setting
        node_ckpt = self.net.get(CKPT, None)
        if node_ckpt is None:
            if ckpt_dir is None:
                ckpt_dir = self.train.get(CKPT_DIR, None)
            node_ckpt = CfgNode({
                DIR: ckpt_dir,
                NAME: ckpt_name,
            })
        else:
            if isinstance(node_ckpt, Config_net_ckpt):
                node_ckpt.set_net_ckpt(ckpt_dir, ckpt_name)
            else:
                if ckpt_dir:
                    node_ckpt[DIR] = ckpt_dir
                if ckpt_name:
                    node_ckpt[NAME] = ckpt_name
        self.net.ckpt = node_ckpt

# a more generic yaml loader without format constraints.
def load_yaml(path):
    from tools.cfgnode import CfgNode
    with open(path) as f:
        yml = yaml.safe_load(f)
    return CfgNode(yml)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="cfg")
    args = parser.parse_args()

    config_loader = Config_loader(args.cfg)
    print(config_loader.get("aha", 5))
    print(config_loader.net.face_feature)
