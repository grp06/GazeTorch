import sys
sys.path.append("../..")

import os
import numpy as np
import cv2
from tqdm import tqdm
from cv.facemask import get_facemaskpoly

root = '/data/gazecapture_3d/face'
dir_target = "/data/gazecapture_3d/facemask"
NUM_DATA=2113373

def batch():
    if not os.path.isdir(dir_target):
        os.mkdir(dir_target)
    for fn in tqdm(sorted(os.listdir(root))):
        path_in = os.path.join(root, fn)
        if not path_in.endswith(".jpg"): continue
        path_out = os.path.join(dir_target, fn[:-4]+".png")
        rgb = cv2.imread(path_in)[:,:,::-1]
        out = get_facemaskpoly(rgb)
        if out is None:
            print("Face not found:", path_in)
            continue
        cv2.imwrite(path_out, out)
    print("Voila.")

if __name__=='__main__':
    batch()