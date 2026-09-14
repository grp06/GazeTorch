import os
import cv2
import numpy as np
from tqdm import tqdm
from cv.segmap import Segmap

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", dest="start", type=int, default=0)
parser.add_argument("-e", dest="end", type=int, default=0)

root = '/data/gazecapture_3d/face'
dir_target = "/data/gazecapture_3d/seg"
seg = Segmap()

NUM_DATA=2113373

def batch(start=0, end=0):
    if end==0 or end>NUM_DATA: end=NUM_DATA
    if not os.path.isdir(dir_target):
        os.mkdir(dir_target)
    fns = sorted(os.listdir(root))
    for i in tqdm(range(start, end)):
        path_in = os.path.join(root, fns[i])
        if not path_in.endswith(".jpg"):
            continue
        path_out = os.path.join(dir_target, fns[i][:-4]+'.png')
        rgb = cv2.imread(path_in)[:,:,::-1]
        out = seg(rgb)
        if out is None:
            print("Face not found:", path_in)
            continue
        cv2.imwrite(path_out, out)
    print("Voila.")

if __name__=='__main__':
    args = parser.parse_args()
    batch(args.start, args.end)