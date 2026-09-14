import torch
import torch.onnx
from torch import Tensor
import numpy as np
import torch.nn.functional as F
import os
import sys
sys.path.append('.')
from model.net import Net_loader
from config.config_loader.config_loader import Config_loader, load_yaml
from tools.cfgnode import CfgNode
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default=None)
parser.add_argument("--slim", type=str, default=False)
parser.add_argument("--model_path", type=str)
parser.add_argument("--ckpt_name", type=str)
parser.add_argument("--output_path", "-o", type=str)


def model_convert(model_path, ckpt_name, cfg_path, output_path, slim_mode=False):

    output_onnx_path = os.path.join(output_path, "onnx_models")
    output_pkl_path = os.path.join(output_path, "pkl_models")

    os.makedirs(output_onnx_path, exist_ok=True)
    os.makedirs(output_pkl_path, exist_ok=True)

    cfg = Config_loader(cfg_path)
    cfg.set_net_ckpt(model_path, ckpt_name)

    # In training yaml, add 'pkl_args' under original 'args' section to mlp head if use CAR loss,
    # 'pkl_args' contains nbins for pitch and yaw, bin size and pitch bias.

    if cfg.net.head.module_output[0][0] != 2:
        if cfg.data.dataset_train.args.get('bin_args', None) is not None:
            nbins_pitch = cfg.data.dataset_train.args.bin_args.get('pitch_nbins', 11)
            nbins_yaw = cfg.data.dataset_train.args.bin_args.get('yaw_nbins', 13)
            bin_size = cfg.data.dataset_train.args.bin_args.get('bin_size', 4.0)
            pitch_bias = cfg.data.dataset_train.args.bin_args.get('pitch_bias', -8.0)
        else:
            nbins_pitch = 15
            nbins_yaw = 17
            bin_size = 4.0
            pitch_bias = -12.0
        pkl_node = CfgNode({
                    'nbins_pitch': nbins_pitch,
                    'nbins_yaw': nbins_yaw,
                    'bin_size': bin_size,
                    'pitch_bias': pitch_bias,
                    'temperature': cfg.train.get('bin_temperature', 1.),
                    'max_out': cfg.train.get('max_out', False),
                    })
        cfg.net.head.args.pkl_args = pkl_node

    # in slim mode, delete other head in cfg file
    if slim_mode:
        for key in list(cfg.net.__dict__.keys()):
            if 'head' in key and key != 'head':
                del cfg.net.__dict__[key]
                cfg.net.output.remove(key)
    print(cfg.net)

    # Load the model, model structure based on "Net" section in config file
    net_loader = Net_loader(cfg.net)
    model = net_loader.model

    model.eval() # Set to evaluation mode

    if slim_mode:
        save_name = model_path.split('/')[-1] + '_ep' + ckpt_name[11:16].lstrip('0') + '_slim'
    else:
        save_name = model_path.split('/')[-1] + '_ep' + ckpt_name[11:16].lstrip('0')

    # save .pkl model here
    torch.save(model, os.path.join(output_pkl_path, save_name + ".pkl"))

    # convert pyth model to onnx model for HQ here
    dummy_input = [torch.randn(1, 3, 224, 224)]

    dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
    torch.onnx.export(model, dummy_input, os.path.join(output_onnx_path, save_name + ".onnx"), verbose=True, 
                    input_names=['input'], output_names=['output']
                    )


if __name__=='__main__':
    args = parser.parse_args()

    cfg_path = args.cfg # yaml file dir
    model_path = args.model_path # 'app/modules/250507_xgaze_bisenetv2_dice0.5_ep9.pkl'
    ckpt_name = args.ckpt_name # "ckpt_epoch_00014.pyth"
    slim_mode = args.slim
    output_path = args.output_path

    model_convert(model_path, ckpt_name, cfg_path, output_path, slim_mode)

