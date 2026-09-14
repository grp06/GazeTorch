import os
from pathlib import Path
import sys
import os, sys
sys.path.append('/store/joelee217/GazeTorch')
from tools.backgroundremover import BackgroundRemover
import sys
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
import concurrent.futures
PATH_GT_DEFAULT = '/data/gazecapture_3d/gt_p17y17_with_conf_v3.npy'
PATH_GT_LITTLE = '/data/gazecapture_3d/gt_little.npy'

def main():
    # replace paths as necessary
    valid_id_path = "/data/gazecapture_3d/valid_ids.npy"

    src_path = "/data/gazecapture_3d/face"
    src_imgs = os.listdir(src_path)

    save_path = "/data/gazecapture_3d/mp_backgrounds"

    # # Check if lmk directory exists, if not create dir
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    valid_ids = np.load(valid_id_path, allow_pickle=True)
    fn = np.load(PATH_GT_DEFAULT, allow_pickle=True).item()['fn']

    valid_fns = [fn[idx] for idx in valid_ids]

    with concurrent.futures.ThreadPoolExecutor(max_workers=72) as executor:
        for _ in executor.map(BackgroundRemover.process_dir_concurrent_mp, tqdm(valid_fns)):
            continue
    
    print(len(valid_fns) == len(os.listdir(save_dir)))

if __name__ == '__main__':
    main()