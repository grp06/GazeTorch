'''
GS = glasses simulator
for image, we save an NPY of 34x2
line 0-15: left eye landmarks (outer contour, not fitting the actual contour)
line 16-31: right eye landmarks
line 32, 33: left/right centroid coordinate
'''
import sys
sys.path.append("../..")

import numpy as np
import cv2
import os
from tqdm import tqdm
from cv.landmark_detector import LandmarkDetector

ID_RIGHT_EYE_OUTER_CONTOUR = [
    464, 413, 441, 442, 443, 444, 445, 342, 446,
    261, 448, 449, 450, 451, 452, 453
    ]

ID_LEFT_EYE_OUTER_CONTOUR = [
    244, 189, 221, 222, 223, 224, 225, 113, 226,
    31, 228, 229, 230, 231, 232, 233
    ]

ID_ALL = ID_LEFT_EYE_OUTER_CONTOUR + ID_RIGHT_EYE_OUTER_CONTOUR # 32x2

ID_RIGHT_TOP = [413, 441, 442, 443, 444, 445, 342]
ID_RIGHT_BOTTOM = [261, 448, 449, 450, 451, 452, 453]
ID_LEFT_TOP = [189, 221, 222, 223, 224, 225, 113]
ID_LEFT_BOTTOM = [31, 228, 229, 230, 231, 232, 233]

ld = LandmarkDetector()

def generate_npy_xgaze(target_dir):
    root = '/data/xgaze/448'
    M = 756540
    target_dir = os.path.join(root, target_dir)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    face_dir = os.path.join(root, 'face_224')
    for i in tqdm(range(M)):
        path_jpg = os.path.join(face_dir, '{}.jpg'.format(i))
        rgb = cv2.imread(path_jpg)[:,:,::-1]
        l2d = ld.landmarks_2D(rgb)

        if l2d is None:
            print("LD failed at #", i)
            continue

        ld_L = l2d[ID_LEFT_EYE_OUTER_CONTOUR, :]
        ld_R = l2d[ID_RIGHT_EYE_OUTER_CONTOUR, :]
        cen_L = np.mean(ld_L, axis=0)[None, :] # (2)=>(1,2)
        cen_R = np.mean(ld_R, axis=0)[None, :]

        matrix = np.concatenate([ld_L, ld_R, cen_L, cen_R], axis=0)

        path_target = os.path.join(target_dir, "{}.npy".format(i))
        np.save(path_target, matrix)

def generate_npy_gc3d(target_dir):
    root = '/data/gazecapture_3d'
    target_dir = os.path.join(root, target_dir)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    face_dir = os.path.join(root, 'face')
    for fn in tqdm(os.listdir(face_dir)):
        if not fn.endswith('.jpg'):
            continue
        path_jpg = os.path.join(face_dir, fn)
        rgb = cv2.imread(path_jpg)[:,:,::-1]
        l2d = ld.landmarks_2D(rgb)

        if l2d is None:
            print("LD failed at #", fn)
            continue

        ld_L = l2d[ID_LEFT_EYE_OUTER_CONTOUR, :]
        ld_R = l2d[ID_RIGHT_EYE_OUTER_CONTOUR, :]
        cen_L = np.mean(ld_L, axis=0)[None, :] # (2)=>(1,2)
        cen_R = np.mean(ld_R, axis=0)[None, :]

        matrix = np.concatenate([ld_L, ld_R, cen_L, cen_R], axis=0)

        path_target = os.path.join(target_dir, "{}.npy".format(fn[:-4]))
        np.save(path_target, matrix)

generate_npy_gc3d('gs_landmarks')