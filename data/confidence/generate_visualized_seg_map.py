from cv.segmap import Segmap
import numpy as np
import cv2
import os

SEG_DIRS = np.load("/data/ninajiang/select_6n/test_save_face_seg_gaze/seg_dirs.npy")

TOTAL_NUM_IMGS = len(SEG_DIRS)


def get_seg_visualized():

    for i in range(TOTAL_NUM_IMGS):
        print("Trial ", i)
        path_seg = SEG_DIRS[i]
        seg = cv2.imread(path_seg, cv2.IMREAD_UNCHANGED)

        demo = Segmap.demo_png(seg)
        # demo = demo[:, :, ::-1]

        cv2.imwrite(os.path.join("/data/ninajiang/select_6n/visualized_seg_demo/", str(i)+".png"), demo)


get_seg_visualized()
print("Images Saved")
