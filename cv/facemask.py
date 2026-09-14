import numpy as np
import cv2

from cv.landmark_detector import LandmarkDetector
ld = LandmarkDetector()
FACEMASK_IDS = [123, 352, 198, 420, 195, 136, 365, 152, 36, 266, 170, 395]

def get_facemaskpoly(rgb):
    l2d = ld.landmarks_2D(rgb)
    if l2d is None: return None
    l2d = l2d.astype(int)
    # points = []
    # for l_id in FACEMASK_IDS:
    #     points.append(l2d[l_id])
    l2d_mask = l2d[FACEMASK_IDS, ...]
    hull = cv2.convexHull(l2d_mask)

    mask = cv2.fillPoly(np.zeros_like(rgb), [hull], (255,255,255))
    return mask