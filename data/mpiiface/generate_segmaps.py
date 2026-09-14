import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from cv.segmap import Segmap

root = '/data/mpiiface/224png'
dir_target = "/data/mpiiface/seg"

seg = Segmap()

if __name__=='__main__':
    n_skipped = 0
    for img_dir in tqdm(glob.glob(root + '/*')):
        img_name = img_dir.split('/')[-1]
        path_out = dir_target + '/' + img_name
        if os.path.isfile(path_out):
            continue
        img = cv2.imread(img_dir)
        rgb = img[:, :, ::-1]
        out = seg(rgb)
        if out is None:
            n_skipped += 1
        else:
            cv2.imwrite(path_out, out)

    print("Num Skipped: {}/{}".format(n_skipped, len(glob.glob(root + '/*'))))
