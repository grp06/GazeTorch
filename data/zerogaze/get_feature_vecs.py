'''
Generate feature vecs through model

args   | example                     | explanation
-n     | "mobile_net"                -> name of numpy file to be saved at output dir 
-c     | "/../ckpt_epoch_00035.pyth" -> path to .pyth model checkpoint
-y     | "/../zerogaze_tsne.yaml"    -> YAML file used for training model
-u     | "vit_b_16_joint"            -> model type for the unigaze model [vit_b_16_joint, vit_h_14_cross_X, vit_h_14_joint] if not using unigaze, leave empty
-o     | "/../..dir/"                -> output directory; will create one if doesn't exist
-s     | 10000                       -> split / specifiy number of images for generation if not specified, will use 'all' samples from dataset
-m     | -m                          -> if -m flag is on, will mirror the images
-t     | -t                          -> if -t flag is on, wil generate tSNE plots based on the feature vectors

~80% faster than using individual call using torch.DataParallel + batching

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
    feature_list = []
    with torch.inference_mode():
        for (X, _) in tqdm(data_loader):
            X = X[0].unsqueeze(0)
            batch_output = model(X.to(device, non_blocking=True))
            feature_list.append(batch_output['face_feature'][0].detach().cpu())
    
    return (torch.stack(feature_list).view(-1, batch_output['face_feature'][0].shape[1]).cpu().numpy())
    
def getPreds_Unigaze(model: nn.Module, data_loader: DataLoader, device) -> np.ndarray:
    res_list = []
    with torch.inference_mode():
        for (X, _) in tqdm(data_loader):
            output = model(X[0])
            batch_output = output[1]
            res_list.append(batch_output.detach().cpu())
    
    return (torch.stack(res_list).view(-1, batch_output.shape[1]).cpu().numpy())


def getFeatureMap(ckpt_path: str = None, 
                  yaml_path: str = None,
                  mirrored: bool = False,
                  split = 'all') -> np.ndarray:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_unigaze = False

    if Path(ckpt_path).exists() and ckpt_path.endswith('.pyth'):
        # load yaml + model architecture
        cfg = Config_loader(yaml_path)   

        ckpt_dir = str(Path(ckpt_path).parent)
        ckpt_name = str(Path(ckpt_path).name)

        cfg.set_net_ckpt(ckpt_dir, ckpt_name)     
        net_loader = Net_loader(cfg.net)
        model = net_loader.model

    elif ckpt_path in ["vit_b_16_joint", "vit_h_14_cross_X", "vit_h_14_joint"]:
        # Assuming we are using external model .pth.tar -> Unigaze
        ckpt_dict = {
            "vit_b_16_joint":  "/data/Zheng/gaze_paper_ckpts/UniGaze/unigaze_b16_joint.pth.tar",
            "vit_h_14_cross_X": "/data/Zheng/gaze_paper_ckpts/UniGaze/unigaze_h14_cross_X.pth.tar",
            "vit_h_14_joint": "/data/Zheng/gaze_paper_ckpts/UniGaze/unigaze_h14_joint.pth.tar"
        }
        model = MAE_Gaze(model_type='vit_b_16') if ckpt_path == "vit_b_16_joint" else  MAE_Gaze(model_type='vit_h_14')
        print('Loading the pre-trained Unigaze model: ', ckpt_path)
        is_unigaze = True
        ckpt = torch.load(ckpt_dict[ckpt_path])
        model.load_state_dict(ckpt['model_state'], strict=True)

    else:
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Model file not found at: {ckpt_path}")
        else:
            raise ValueError(
                f"Unsupported model format: {Path(ckpt_path).suffix}. "
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

    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pyth)")
    
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
    
    parser.add_argument("-t", "--tnse",  action="store_true", default=False,
                        help="Decides to whether to generate a tSNE plot based on the feature vectors")

    args = parser.parse_args()

    checkpoint_path = args.unigaze if args.checkpoint is None else args.checkpoint 

    res_v, res_g, res_m = getFeatureMap(
                                    ckpt_path=checkpoint_path, 
                                    yaml_path=args.yaml,
                                    mirrored=args.mirror, 
                                    split=int(args.split) if type(args.split) == str else 'all'
                                )
    

    os.makedirs(args.output, exist_ok=True)

    np.save(Path(args.output) / f"{args.name}_features_vanilla.npy", res_v)
    np.save(Path(args.output) / f"{args.name}_features_glasses.npy", res_g)
    np.save(Path(args.output) / f"{args.name}_features_masks.npy", res_m)

    print(f"Feature Maps saved at dir: {args.output}")

    if args.tnse:
        print("[tSNE]: generating tSNE ... ")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        # generate tSNE plot and save
        v_labels = ['vanilla'] * len(res_v)
        g_labels = ['glasses'] * len(res_g)
        m_labels = ['masks'] * len(res_m)

        features = np.concatenate((res_v, res_g, res_m), axis=0)
        labels = v_labels + g_labels + m_labels
        labels_np = np.array(labels)

        perplexity = 100 # could be a args?

        tsne = TSNE(
            n_components=2, 
            learning_rate='auto', 
            init='random', 
            perplexity=perplexity,  
        )

        X_embedded = tsne.fit_transform(features)
        fig, ax = plt.subplots(figsize=(12, 8))

        unique_labels = ['vanilla', 'glasses', 'masks']
        colors = ['skyblue', 'salmon', 'lime']

        for i, label in enumerate(unique_labels):
            indices = np.where(labels_np == label)[0]
    
            if len(indices) == 0:
                print(f"Warning: No data found for label '{label}'")
                continue
            
            ax.scatter(
                X_embedded[indices, 0], 
                X_embedded[indices, 1], 
                c=colors[i], 
                label=label,
                alpha=0.5,
                s=15,       # Size of points
                edgecolors='none'
            )

        ax.set_title(f"2D t-SNE Projection of Gaze Features ({args.name}) - Perplexity: {perplexity}", fontsize=15)
        ax.set_xlabel("t-SNE dimension 1") 
        ax.set_ylabel("t-SNE dimension 2")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title="Categories", loc='best')

        plt.tight_layout()
        plt.savefig(Path(args.output) / f'{args.name}_p{perplexity}.png')
        print(f"tSNE plot saved at dir: {args.output}")


