import torch
import os
import sys
sys.path.append('.')
from model.net import Net_loader
from config.config_loader.config_loader import Config_loader, load_yaml
from tools.cfgnode import CfgNode
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default=None)
parser.add_argument("--model_path", type=str)
parser.add_argument("--ckpt_name", type=str)
parser.add_argument("--target_dir", "-o", type=str)

# slim: if True, remove aux branches.
parser.add_argument("--slim", action='store_true')
parser.set_defaults(slim=False)


def model_convert(model_path, ckpt_name, cfg_path, target_dir, slim_mode=False):
    os.makedirs(target_dir, exist_ok=True)

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
                    })
        print("[Converter]pkl_args:", pkl_node)
        cfg.net.head.args.pkl_args = pkl_node

    # in slim mode, delete other head in cfg file
    if slim_mode:
        for key in list(cfg.net.__dict__.keys()):
            if 'head' in key and key != 'head' and key in cfg.net.output:
                del cfg.net.__dict__[key]
                cfg.net.output.remove(key)

    # Load the model, model structure based on "Net" section in config file
    net_loader = Net_loader(cfg.net)
    model = net_loader.model

    model.eval() # Set to evaluation mode

    from tools.torch_ops import count_params
    print("[Converter]Model Param #:", count_params(model))

    save_name = model_path.split('/')[-1] + '_ep' + ckpt_name[11:16].lstrip('0')

    # save .pkl model here
    target_path = os.path.join(target_dir, save_name + ".pkl")
    torch.save(model, target_path)
    print("[Converter]PKL stored at:", target_path)

if __name__=='__main__':
    args = parser.parse_args()

    cfg_path = args.cfg # yaml file dir
    model_path = args.model_path # 'app/modules/250507_xgaze_bisenetv2_dice0.5_ep9.pkl'
    ckpt_name = args.ckpt_name # "ckpt_epoch_00014.pyth"
    target_dir = args.target_dir
    slim_mode = args.slim

    model_convert(model_path, ckpt_name, cfg_path, target_dir, slim_mode)