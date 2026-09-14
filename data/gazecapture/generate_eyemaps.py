import sys
sys.path.append("../..")

import cv2
import os
import numpy as np
from cv.iris_detector import MPIris, iris_cdf_with_search
from tqdm import tqdm

root = '/data/gazecapture_3d'
dir_iris_model = '/root/GazeTorch/cv/modules'

def test():
    id = MPIris(dir_iris_model)
    for side in ('L','R'):
        dir_source = os.path.join(root, 'eye_{}'.format(side))
        # dir_eyemap = os.path.join(root, 'eye_{}_eyemap2'.format(side))
        # dir_irismap = os.path.join(root, 'eye_{}_irismap2'.format(side))
        dir_eyemap = os.path.join(root, 'eye_{}_x'.format(side))
        dir_irismap = os.path.join(root, 'eye_{}_x'.format(side))
        if not os.path.isdir(dir_eyemap): os.mkdir(dir_eyemap)
        if not os.path.isdir(dir_irismap): os.mkdir(dir_irismap)
        for c,fn in tqdm(enumerate(os.listdir(dir_source))):
            if not fn.endswith(".jpg"): continue
            path = os.path.join(dir_source, fn)
            # if not os.path.isfile(path):
                # continue
            path_eyemap = os.path.join(dir_eyemap, 'E'+fn)
            path_irismap = os.path.join(dir_irismap, 'I'+fn)
            bgr = cv2.imread(path)
            if side=='R':
                bgr = cv2.flip(bgr, 1)

            rgb = bgr[:,:,::-1]
            Y,mask,cx,cy,poly = iris_cdf_with_search(rgb, id)
            if side=='R':
                mask = cv2.flip(mask, 1)
                poly = cv2.flip(poly, 1)
            
            cv2.imwrite(path_eyemap, poly)
            cv2.imwrite(path_irismap, mask)
            cv2.imwrite(os.path.join(dir_eyemap, fn), bgr)
            if c==2000: break
        break
    print("Voila.")

def generate():
    id = MPIris(dir_iris_model)
    for side in ('L','R'):
        dir_source = os.path.join(root, 'eye_{}'.format(side))
        dir_eyemap = os.path.join(root, 'eye_{}_eyemap2'.format(side))
        dir_irismap = os.path.join(root, 'eye_{}_irismap2'.format(side))
        if not os.path.isdir(dir_eyemap): os.mkdir(dir_eyemap)
        if not os.path.isdir(dir_irismap): os.mkdir(dir_irismap)
        for fn in tqdm(os.listdir(dir_source)):
            if not fn.endswith(".jpg"): continue
            path = os.path.join(dir_source, fn)
            # if not os.path.isfile(path):
                # continue
            path_eyemap = os.path.join(dir_eyemap, fn)
            path_irismap = os.path.join(dir_irismap, fn)
            if os.path.isfile(path_eyemap):
                continue
            bgr = cv2.imread(path)
            if side=='R':
                bgr = cv2.flip(bgr, 1)

            rgb = bgr[:,:,::-1]
            Y,mask,cx,cy,poly = iris_cdf_with_search(rgb, id)
            if side=='R':
                mask = cv2.flip(mask, 1)
                poly = cv2.flip(poly, 1)
            
            cv2.imwrite(path_eyemap, poly)
            cv2.imwrite(path_irismap, mask)
    print("Voila.")

def to_binary(suffix='0.2', threshold=0.2):
    def to_binary_sub(dir_source, dir_target, threshold=0.2):
        for fn in tqdm(os.listdir(dir_source)):
            if not fn.endswith(".jpg"): continue
            path = os.path.join(dir_source, fn)
            path_out = os.path.join(dir_target, fn)

            bgr = cv2.imread(path).astype(float)/255
            bgr_out = (bgr>threshold).astype(float)*255
            cv2.imwrite(path_out, bgr_out)
        print("Voila.")

    dirs = ("eye_L_eyemap", "eye_L_irismap", "eye_R_eyemap", "eye_R_irismap")
    for dn in dirs:
        print(dn)
        dir_source = os.path.join(root, dn)
        dir_target = os.path.join(root, dn+suffix)
        if not os.path.isdir(dir_target):
            os.mkdir(dir_target)
        to_binary_sub(dir_source, dir_target, threshold=threshold)

if __name__ == '__main__':
    generate()
    # to_binary()