import sys
sys.path.append("../..")

import os
import numpy as np
import cv2
from tqdm import tqdm
from cv.crop_eye import crop_eye
from cv.landmark_detector import LandmarkDetector

import argparse
M = 756540
parser = argparse.ArgumentParser()
parser.add_argument("-s", dest="start", type=int, default=0)
parser.add_argument("-e", dest="end", type=int, default=M)

dir_face = '/data/xgaze/448/face'
target = '/data/xgaze/448/eye_{}_0610'
#cam_ids = [0, 1, 2, 3, 4, 5, 6, 7, 9]
cam_ids = [9]
out_size = (150, 90)
pad = (0.5, 0.5)

ld = LandmarkDetector()

def batch(start, end):
    for c in ('L','R'):
        d = target.format(c)
        if not os.path.isdir(d):
            os.mkdir(d)
    for i in tqdm(range(start, end)):
        if i%18 not in cam_ids: continue
        path_face = os.path.join(dir_face, "%d.jpg" % i)
        if not os.path.isfile(path_face): continue
        rgb = cv2.imread(path_face)[:,:,::-1]

        l2d_raw = ld.landmarks_2D(rgb)
        if l2d_raw is None: continue

        _, eyes = crop_eye(rgb, landmarks=l2d_raw, size=out_size, pad=pad)
        if eyes is None:
            print("Invalid eye patches @", i)
            continue
        for j, c in enumerate(('L', 'R')):
            path_eye = os.path.join(target.format(c), '%d.jpg'%i)
            cv2.imwrite(path_eye, eyes[j][:,:,::-1])
    print("Voila.")

if __name__=='__main__':
    args = parser.parse_args()
    batch(start=args.start, end=args.end)