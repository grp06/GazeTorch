import sys
sys.path.append('../..')

import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from cv.terminator import Terminator


root = '/data/gaze360/face_224'
dir_target = "/data/gaze360/facemap"

T = Terminator("../../cv/modules")

if __name__=='__main__':
    n_total = len(glob.glob(root + '/*'))
    n = 0
    for img_dir in glob.glob(root + '/*'):
        print('{} out of {}'.format(n, n_total), end='\r')
        n += 1
        img_name = img_dir.split('/')[-1]
        path_out = dir_target + '/' + img_name
        if os.path.isfile(path_out):
            continue
        img = cv2.imread(img_dir)
        rgb = img[:, :, ::-1]
        out = T.run(rgb)
        if out is None:
            print('OOPS ...')
        else:
            cv2.imwrite(path_out, out)
