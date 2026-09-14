
# this is the offical launcher of the program, please go over the wiki/readme to learn how to use it

'''
GazeTorch launcher, by Zhenhao and Justin Cai.

Example usage:

# To train (single-gpu)
CUDA_VISIBLE_DEVICES="4" python gazetorch.py -i config/.../xxx.yaml --single_gpu

# To train (multi-gpu)
CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 --master_port 00000 gazetorch.py --cfg_path config/.../xxx.yaml

# To eval a TFLITE model
python gazetorch.py -m eval -i /data/.../*.tflite -d config/config_lib/eval/gaze360.yaml -f tf

# To eval a PyTorch ckpt & save results
python gazetorch.py -m eval -f torch -i config/.../xxx.yaml -d config/config_lib/eval/ddd.yaml -o temp_results/result.npy --load_ckpt_dir /data/...(ckpt_dir)/ --load_ckpt_name ckpt_epoch_00000.pyth
'''

import torch
import argparse
import os

import logging
import sys

from config.config_loader.config_loader import Config_loader, load_yaml
from train.config_handler import *
from tools.cfgnode import CfgNode

parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", "--local_rank", type=int) # every sub-process will get this arg from main process
parser.add_argument("-i", "--cfg_path", type=str) # every sub-process will get this arg from main process

# mode (-m): train/eval/net_debug
parser.add_argument("-m", "--mode", type=str, default='train', help="Mode: train/eval/net_debug")

# load_ckpt_dir, load_ckpt_name: load from a specified ckpt.
# if load_ckpt_name is None, then load the latest ckpt in `load_ckpt_dir`
parser.add_argument("--load_ckpt_dir", type=str, default=None,
help="If specified, load ckpt from this dir.")
parser.add_argument("--load_ckpt_name", type=str, default=None,
help="If specified, load this ckpt (typically a .pth file). If not, load the latest ckpt.")

# =======TRAIN MODE ARGS=======
# Refer to train/config_handler.py

# =======EVAL MODE ARGS=======
# dataset (-d): the target domain dataset.
parser.add_argument("-d", "--cfg_dataset", type=str, default=None,
help="For eval mode only. The target domain dataset.")

# output path (-o): if specified, save model inference result at the path.
parser.add_argument("-o", "--output_path", type=str, default=None,
help="For eval mode only. If specified, save model inference result (gaze) at the path.")

# framework (-f): for eval mode only. can be "torch", "tf" or "tf_old".
# tf_old is for for XGaze tflite models before 2021:
    # mean=(0.485, 0.456, 0.406)
    # std=(0.229, 0.224, 0.225)
parser.add_argument("-f", "--framework", type=str, default='torch',
help="For eval mode only. Specifying model type, can be torch/tf/tf_old.")

# result_module, result_tensor_id: if not None, specify the output module & Tensor.
# Example: for output=[head, head_seg, head_ssl]
# then, to output gaze, let result_module='head', result_tensor_id=0
# user can also specify other output modules & Tensors. But please be cautious about loss computing in this case.
parser.add_argument("--result_module", type=str, default=None,
help="If specified, the output module name; else, take the first module in `cfg.net.output`.")
parser.add_argument("--result_tensor_id", type=int, default=0,
help="If specified, the Tensor index in `result_module`.")

# skip_eval: if True, skip computing the loss, just save the output. 
parser.add_argument("--skip_eval", action='store_true')
parser.set_defaults(skip_eval=False)

# device: "cpu", or an int specifying GPU ID
parser.add_argument("--device", type=str, default='0',
help="\"cpu\", or an int specifying GPU ID")

# seg: if True, toggle "demo segmentation" mode.
parser.add_argument("--seg", action='store_true')
parser.set_defaults(seg=False)

# step: for "--seg" only. if set, demonstrate Images #(step*i) only.
parser.add_argument("--step", type=int, default=1)

# =======NET_DEBUG MODE=======
# net_debug: if True, just load network and return (for debug)

if __name__ == "__main__":
    add_arguments(parser)
    args = parser.parse_args()

    if args.mode == 'train':
        cfg = Config_loader(args.cfg_path)
        cfg = modify_cfg(cfg, args)
        
        print("[GazeTorch]TRAIN MODE...")
        local_rank = int(os.environ["LOCAL_RANK"])
        print("[GazeTorch]NCCL initialized. local_rank =", local_rank)

        if cfg.train.gpus != 'cpu':
            torch.cuda.set_device(local_rank)
            if isinstance(cfg.train.gpus, list) and len(cfg.train.gpus)>1:
                import torch.distributed as dist
                dist.init_process_group(backend='nccl')

        logging.basicConfig(level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
        logger = logging.getLogger(__name__)
        
        # let main process print only (by Justin)
        # DEBUG: send non-master ranks to per-rank files (NOT /dev/null) so their
        # tracebacks / crash dumps survive. The old /dev/null discarded every worker error.
        if local_rank not in [0, -1]:
            import faulthandler
            f = open(f"/tmp/gaze_rank{local_rank}.log", "w", buffering=1)  # line-buffered
            sys.stdout = f
            sys.stderr = f
            faulthandler.enable(file=f, all_threads=True)  # also dump on segfault/abort

        if cfg.train.get("cv", False):
            from train.trainer_crossval import Trainer
        else:
            from train.trainer import Trainer
        cfg.set_net_ckpt(args.load_ckpt_dir, args.load_ckpt_name)
        t = Trainer(cfg)
        t.train()

    elif args.mode == 'net_debug':
        print("[GazeTorch]NETWORK DEBUG MODE...")
        cfg = Config_loader(args.cfg_path)

        from model.net import Net_loader
        net_loader = Net_loader(cfg.net)
        model = net_loader.model
        model.eval()

        # prepare an input and try forward pass
        input_shape = cfg.net.input
        Xs = []
        for shape in input_shape:
            shape = [2] + shape
            Xs.append(torch.zeros(shape))
        print(f"[GazeTorch]Input:", Xs[0].shape)
        Ys = model(Xs)

        for module_name in cfg.net.output:
            print(f"[GazeTorch]Output: {module_name}")
            ys = Ys[module_name]
            for j,y in enumerate(ys):
                print(f"Output Tensor #{j}'s shape = {y.shape}")
        print("[GazeTorch]Voila. Network passes test.")

    elif args.mode == 'eval':
        print("[GazeTorch]EVAL MODE...")
        from config.config_loader.config_data import Config_data_dataset
        yml_dataset = load_yaml(args.cfg_dataset)
        cfg_dataset = Config_data_dataset(yml_dataset, 'data')
        path_target = args.output_path

        if args.framework == 'torch':
            if args.load_ckpt_name.endswith('.pyth'):
                cfg = Config_loader(args.cfg_path)
                cfg.set_net_ckpt(args.load_ckpt_dir, args.load_ckpt_name)
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
                from tools.tester import Tester
                t = Tester(cfg=cfg, cfg_dataset=cfg_dataset, path_target=path_target,
                    device=args.device,
                    result_module=args.result_module, result_tensor_id=args.result_tensor_id)

                if args.seg:
                    t.demo_segmentation(args.step)
                else:
                    t.run(args.skip_eval)
            else:
                from tools.tester import Tester_pkl
                t = Tester_pkl(model_path=os.path.join(args.load_ckpt_dir, args.load_ckpt_name), cfg_dataset=cfg_dataset, path_target=path_target,
                    device=args.device,
                    result_module=args.result_module, result_tensor_id=args.result_tensor_id)
                t.run(args.skip_eval)

        else:
            from tools.tester_tf import Tester_tflite
            classic_normalization = args.framework == 'tf_old'
            t = Tester_tflite(tflite_model_path=args.cfg_path, cfg_dataset=cfg_dataset, classic_normalization=classic_normalization,
                    path_target=path_target)
            t.run()