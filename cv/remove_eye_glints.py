import numpy as np
import cv2
from skimage import morphology

def remove_eye_glint(rgb, radius=3):
    t = np.zeros((radius*2+5, radius*2+5))
    t[2:radius*2+3, 2:radius*2+3] = morphology.disk(radius)
    gauss_kernel = cv2.GaussianBlur(t, (radius*2+1, radius*2+1), 2.)
    rgb = cv2.morphologyEx(rgb, cv2.MORPH_OPEN, gauss_kernel)
    return rgb