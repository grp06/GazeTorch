import sys
sys.path.append('../..')

import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from cv.terminator import Terminator


root = '/data/eyediap/normalized_face_224_cs'
dir_target = "/data/eyediap/facemap_cs"

T = Terminator("../../cv/modules")

if __name__=='__main__':
    for img_dir in tqdm(glob.glob(root + '/*')):
        img = cv2.imread(img_dir)
        img_name = img_dir.split('/')[-1]
        path_out = dir_target + '/' + img_name
        rgb = img[:, :, ::-1]
        out = T.run(rgb)
        if out is None:
            print('OOPS ...')
        else:
            cv2.imwrite(path_out, out)
