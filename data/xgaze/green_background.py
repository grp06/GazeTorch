import os
import cv2
import numpy as np
from tqdm import tqdm

class Green_Background_Remover:
    L_GREEN = np.array([70., 0.5, 0.])
    U_GREEN = np.array([170., 1., 1.])

    @staticmethod
    def process_rgb(rgb):
        rgb = rgb.astype(np.float32)
        rgb /= np.amax(rgb)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, Green_Background_Remover.L_GREEN, Green_Background_Remover.U_GREEN)
        mask = np.stack([mask,mask,mask], axis=2) # 0~255
        return mask

    @staticmethod
    def process_dir(source_dir, target_dir):
        for fn in tqdm(os.listdir(source_dir)):
            if not fn.endswith(".jpg"): continue
            path = os.path.join(source_dir, fn)
            rgb = cv2.imread(path)[:,:,::-1]
            mask = Green_Background_Remover.process_rgb(rgb)
            
            if mask is None:
                print("[GBR]Empty mask:", fn)
                continue

            path_target = os.path.join(target_dir, fn)
            cv2.imwrite(path_target, mask)
        print("Voila.")

source_dir = "/data/xgaze/448/face_224"
target_dir = "/data/xgaze/448/green"
Green_Background_Remover.process_dir(source_dir, target_dir)