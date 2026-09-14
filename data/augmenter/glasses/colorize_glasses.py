"""
[Joe Lee]
* Uses the red glasses mask and converts it into the 7 colors needed for augmentation.
* First convert to HSV, then use the dict_colors to shift the hues accordingly.
* Need to combine the final masks/color files later on server
"""

import cv2, os
import numpy as np
import shutil
from tqdm import tqdm

dict_colors = {
    'red': 0,
    'yellow': 40,
    'green': 90,
    'cyan': 180,
    'blue': -120,
    'pink': -60
}

TEMPDIR = "/data/Glasses_v5/temp_mask_zhenhao2/png/"
SAVEDIR = "/data/Glasses_v5/"

colors_dir = "/data/Glasses_v5/glasses_colors_zhenhao2"
masks_dir = "/data/Glasses_v5/glasses_masks_zhenhao2"

os.makedirs(colors_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

for fn in tqdm(os.listdir(TEMPDIR)):

    src_path = os.path.join(TEMPDIR, fn)
    type = fn.split('.png')[0].split('_')[-1]

    if type == 'mask':
        # move file to masks_dir
        dst_path = os.path.join(masks_dir, fn)
        shutil.copy2(src_path, dst_path) 
    elif type == 'black': 
        dst_path = os.path.join(colors_dir, fn)
        shutil.copy2(src_path, dst_path) 
    else: # red 
        glasses = cv2.imread(src_path)
        img_bgr = glasses[:,:,:3]

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        hue = img_hsv[:,:,0].astype(np.int32)

        for color_name, shift in dict_colors.items():
            new_hue = (hue + shift/2) % 180
            img_hsv_new = img_hsv.copy()
            img_hsv_new[:,:,0] = new_hue

            img_rgb = cv2.cvtColor(img_hsv_new, cv2.COLOR_HSV2BGR)

            cv2.imwrite(f'{colors_dir}/{fn.split("_red")[0]}_{color_name}.png', 
                        img_rgb
                        )




