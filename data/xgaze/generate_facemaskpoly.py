import sys
sys.path.append("../..")

import os
import numpy as np
import cv2
from tqdm import tqdm
from cv.facemask import get_facemaskpoly

root = '/data/xgaze/448/face_224'
dir_target = "/data/xgaze/448/facemask_224"
NUM_DATA=756540

def test():
    for i in tqdm(range(1000)):
        path_in = os.path.join(root, "%d.jpg"%i)
        path_out = os.path.join(dir_target, "%d.png"%i)
        # path_out_v = os.path.join(dir_target, "%d.png"%i)
        rgb = cv2.imread(path_in)[:,:,::-1]
        out = get_facemaskpoly(rgb)
        # out_v = T.visualize()
        if out is None:
            # out_v = np.ones((224,224,3)) * 255
            print(i, i%18)
            continue
        cv2.imwrite(path_out, out)
        # cv2.imwrite(path_out_v, out_v)
    print("Voila.")

def batch(start=0, end=0):
    if end==0 or end>NUM_DATA: end=NUM_DATA
    if not os.path.isdir(dir_target):
        os.mkdir(dir_target)
    for i in tqdm(range(start, end)):
        path_in = os.path.join(root, "%d.jpg"%i)
        path_out = os.path.join(dir_target, "%d.png"%i)
        rgb = cv2.imread(path_in)[:,:,::-1]
        out = get_facemaskpoly(rgb)
        if out is None:
            print(i, i%18)
            continue
        cv2.imwrite(path_out, out)
    print("Voila.")

if __name__=='__main__':
    batch()