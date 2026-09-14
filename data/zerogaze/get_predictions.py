'''
Generate NPY file with gaze (P,Y) results through model

args   | example                     | explanation
-n     | "mobile_net"                -> name of numpy file to be saved at output dir 
-i     | "/../.._ep35_slim.pkl"      -> PKL or PTH model path used 
-y     | "/../.._ep35_slim.pth"      -> YAML file used for trained model (if using PTH file format)
-u     | "vit_b_16_joint"            -> model type for the unigaze model [vit_b_16, vit_h_14_cross_X, vit_h_14_joint] if not using unigaze, leave empty
-o     | "/../..dir/"                -> output directory; will create one if doesn't exist
-s     | 10000                       -> split / specifiy number of images for generation if not specified, will use 'all' samples from dataset
-m     | -m                          -> if -m flag is on, will mirror the images

~80% faster than using individual call using torch.DataParallel + batching

[1/16/2026: Documentation]

As default, our model [head] (mlp) outputs a 22-d vector.

However, when we convert the .pyth (ckpt) to .pkl, we run get_attr() in the forward call at ./model/head/mlp.py 
as we specify the pkl_args = True, the model returns the 2-d (p/y) output instead.

Therefore, .pkl models do not require a separate class for getting pitch/yaw outputs from the model.

However, also note that .pkl models are fixed in model structure, therefore when we load the model, we can't make changes to the model architecture;
for example, we can't attach other modules or simply extract the output of the backbone with .pkl models.

This is also why we load models in .pyth format for the ./data/zerogaze/get_feature_vecs.py as well.

To output the 2-d(p/y) values, we attach a new module from ./model/head/regressor.py and its 'Regressor' class, 
which does the same functionality as the get_attr() part of MLP. 

To do so, we can specify a regressor and take the inputs from the head in the training .yaml file as follows:

net:
  output: [regressor, head]
  ...
  regressor:
    module_input:
      head: all
    module_output: [[2]]
    type: regressor

This gives the flexibility of obtaining both outputs the 22-d vector and 2-d vector from the model.

However, note that in ./train/loss/car.py, these 2 vectors are available as the L1 loss is implemented as part of CAR loss.
'''

import sys, os
sys.path.insert(0, os.getcwd())
from data.dataset.zerogaze_masks import Dataset_ZeroGaze_Masks
from data.dataset.zerogaze_vanilla import Dataset_ZeroGaze_Vanilla
from data.dataset.zerogaze_glasses import Dataset_ZeroGaze_Glasses
from cv.modules.UniGaze.models.vit.mae_gaze import MAE_Gaze
from config.config_loader.config_loader import Config_loader
from model.net import Net_loader

import numpy as np
import torch
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

def getPreds(model: nn.Module, data_loader: DataLoader, device) -> np.ndarray:
    res_list = []
    with torch.inference_mode():
        for (X, _) in tqdm(data_loader):
            X = X[0].unsqueeze(0)
            batch_output = model(X.to(device, non_blocking=True))
            res_list.append(batch_output['head'][0].detach().cpu())
    
    return(torch.stack(res_list).view(-1, 2).cpu().numpy())

def getPreds_pyth(model: nn.Module, data_loader: DataLoader, device) -> np.ndarray:
    res_list = []
    with torch.inference_mode():
        for (X, _) in tqdm(data_loader):
            X = X[0].unsqueeze(0)
            batch_output = model(X.to(device, non_blocking=True))
            res_list.append(batch_output['regressor'][0].detach().cpu())
    
    return(torch.stack(res_list).view(-1, 2).cpu().numpy())

def getPreds_Unigaze(model: nn.Module, data_loader: DataLoader, device) -> np.ndarray:
    res_list = []
    with torch.inference_mode():
        for (X, _) in tqdm(data_loader):
            output = model(X[0])
            batch_output = output[0]['pred_gaze']
            res_list.append(batch_output.detach().cpu())
    
    return(torch.stack(res_list).view(-1, 2).cpu().numpy())


def getGaze(model_path: str = None, 
            yaml_path: str = None,
            mirrored: bool = False,
            split = 'all') -> np.ndarray:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_unigaze = False
    is_pyth = False

    if Path(model_path).exists() and model_path.endswith('.pkl'):
        # PKL format files loading
        model = torch.load(model_path, weights_only=False)
    
    elif Path(model_path).exists() and model_path.endswith('.pyth'):
        is_pyth = True
        # .pth format files loading
        cfg = Config_loader(yaml_path)   

        ckpt_dir = str(Path(model_path).parent)
        ckpt_name = str(Path(model_path).name)

        cfg.set_net_ckpt(ckpt_dir, ckpt_name)     
        net_loader = Net_loader(cfg.net)
        model = net_loader.model

    elif model_path in ["vit_b_16_joint", "vit_h_14_cross_X", "vit_h_14_joint"]:
        is_unigaze = True
        # Assuming we are using external model .pth.tar -> Unigaze
        ckpt_dict = {
            "vit_b_16_joint":  "/data/Zheng/gaze_paper_ckpts/UniGaze/unigaze_b16_joint.pth.tar",
            "vit_h_14_cross_X": "/data/Zheng/gaze_paper_ckpts/UniGaze/unigaze_h14_cross_X.pth.tar",
            "vit_h_14_joint": "/data/Zheng/gaze_paper_ckpts/UniGaze/unigaze_h14_joint.pth.tar"
        }
        model = MAE_Gaze(model_type='vit_b_16') if model_path == "vit_b_16_joint" else  MAE_Gaze(model_type='vit_h_14')
        print('Loading the pre-trained Unigaze model: ', model_path)
        ckpt = torch.load(ckpt_dict[model_path])
        model.load_state_dict(ckpt['model_state'], strict=True)
    else:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        else:
            raise ValueError(
                f"Unsupported model format: {Path(model_path).suffix}. "
                "Expected '.pyth' or specify model for UniGaze with -u ."
            )

    # Run through all three datasets 

    ds_args = {
        'split': split, # either 'train', 'test', 'all' or int: number of samples wanted)
        'return_tensor': True,
        'inputs': ['face'], 
        'labels': []
    }
    dataloader_kwargs = {
        "batch_size": 64,
        "num_workers": 32,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": True
    }
    # Load all three datasets (Only Images) | load the mirrored dataset if specified
    zerogaze_v = Dataset_ZeroGaze_Vanilla(**ds_args, mirrored=True) if mirrored else Dataset_ZeroGaze_Vanilla(**ds_args, mirrored=False)
    zerogaze_g = Dataset_ZeroGaze_Glasses(**ds_args, mirrored=True) if mirrored else Dataset_ZeroGaze_Glasses(**ds_args, mirrored=False)
    zerogaze_m = Dataset_ZeroGaze_Masks(**ds_args, mirrored=True) if mirrored else Dataset_ZeroGaze_Masks(**ds_args, mirrored=False)

    loader_v = DataLoader(zerogaze_v, **dataloader_kwargs)
    loader_g = DataLoader(zerogaze_g, **dataloader_kwargs)
    loader_m = DataLoader(zerogaze_m, **dataloader_kwargs)
    
    model = nn.DataParallel(model).to(device)
    model.eval()

    if is_unigaze:
        res_v = getPreds_Unigaze(model, loader_v, device=device)
        res_g = getPreds_Unigaze(model, loader_g, device=device)
        res_m = getPreds_Unigaze(model, loader_m, device=device)
    elif is_pyth:
        res_v = getPreds_pyth(model, loader_v, device=device)
        res_g = getPreds_pyth(model, loader_g, device=device)
        res_m = getPreds_pyth(model, loader_m, device=device)
    else:
        res_v = getPreds(model, loader_v, device=device)
        res_g = getPreds(model, loader_g, device=device)
        res_m = getPreds(model, loader_m, device=device)

    return res_v, res_g, res_m

if __name__ == "__main__":
    # Parser 
    parser = argparse.ArgumentParser(description="Return NPY of Model Predictions for ZeroGaze(V, G, M)")

    parser.add_argument("-n", "--name", type=str, default="model",
                        help="model name used for printing/display")

    parser.add_argument("-i", "--input", type=str, default=None,
                        help="Path to model file (.pkl/.pth.tar)")
    
    parser.add_argument("-y", "--yaml", type=str, default=None,
                        help="Path to yaml file (.yaml)")
    
    parser.add_argument("-u", "--unigaze", type=str, default=None,
                        help="Unigaze model type")

    # Output Directory
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory for predictions")

    parser.add_argument("-m", "--mirror", action="store_true", default=False,
                        help="Decides whether to produce data on mirrored images")

    parser.add_argument("-s", "--split", default=None,
                        help="Number of samples per output NPY file")

    args = parser.parse_args()
    
    
    model_path = args.unigaze if args.input is None else args.input 

    res_v, res_g, res_m = getGaze(
                                    model_path=model_path, 
                                    yaml_path=args.yaml,
                                    mirrored=args.mirror, 
                                    split=int(args.split) if type(args.split) == str else 'all'
                                )

    os.makedirs(args.output, exist_ok=True)
    np.save(Path(args.output) / f"{args.name}_vanilla.npy", res_v)
    np.save(Path(args.output) / f"{args.name}_glasses.npy", res_g)
    np.save(Path(args.output) / f"{args.name}_masks.npy", res_m)

    print(f"Files saved at : {args.output}")