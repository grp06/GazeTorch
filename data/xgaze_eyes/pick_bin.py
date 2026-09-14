'''
A toy function that generates random samples from 5x5 gaze bins.
Results are saved at $root/bin_samples
'''

import os
import numpy as np
from shutil import copyfile

from sampler import XGaze_2D_Sampler

root = '/data/xgaze/448'
npy_path = os.path.join(root, 'single_eye_p17y17.npy')
dir_out = os.path.join(root, "bin_samples")
if not os.path.isdir(dir_out):
    os.mkdir(dir_out)

D = np.load(npy_path, allow_pickle=True).item()
S_L = XGaze_2D_Sampler(D, split='val', side='L')
S_R = XGaze_2D_Sampler(D, split='val', side='R')
dir_img_L = os.path.join(root, "eye_L_aug_0415")
dir_img_R = os.path.join(root, "eye_R_aug_0415")
bins = (4,6,8,10,12)

n = 50

for p in bins:
    for y in bins:
        dn = os.path.join(dir_out, "p{}y{}".format(p, y))
        print(dn)
        if not os.path.isdir(dn):
            os.mkdir(dn)
        for side in ('L', 'R'):
            S = eval("S_{}".format(side))
            dir_img = eval("dir_img_{}".format(side))
            ids = np.random.permutation(S.bins[p][y])[:n]
            for i in range(n):
                img_id = S.D['id'][ids[i]]
                path_source = os.path.join(dir_img, "{}.jpg".format(img_id))
                path_target = os.path.join(dn, "{}{:02}_{}.jpg".format(side, i, img_id))
                copyfile(path_source, path_target)

print("Voila.")