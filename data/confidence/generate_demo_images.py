from cv.segmap import Segmap
import numpy as np
import cv2
import os

FACE_DIRS = np.load("/data/ninajiang/select_6n/test_save_face_seg_gaze/face_dirs.npy")
SEG_DIRS = np.load("/data/ninajiang/select_6n/test_save_face_seg_gaze/seg_dirs.npy")
GAZE = np.load("/data/ninajiang/select_6n/test_save_face_seg_gaze/gaze.npy")

TOTAL_NUM_IMGS = len(FACE_DIRS)


def get_seg_face_image():

    for i in range(TOTAL_NUM_IMGS):
        print("Trial: ", i)
        path_face = FACE_DIRS[i]
        face = cv2.imread(path_face, cv2.IMREAD_UNCHANGED)
        path_seg = SEG_DIRS[i]
        seg = cv2.imread(path_seg, cv2.IMREAD_UNCHANGED)

        demo = Segmap.demo(face, seg)
        # demo = demo[:, :, ::-1]

        cv2.imwrite(os.path.join("/data/ninajiang/select_6n/face_seg_demo/", str(i)+".png"), demo)


get_seg_face_image()
print("Images Saved")
