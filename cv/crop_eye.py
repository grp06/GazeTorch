import numpy as np
import cv2
from tools.utils import image_preprocess

ids = [[33,246,161,160,159,158,157,173,133,7,163,144,145,153,154,155],
      [362,398,384,385,386,387,388,466,263,382,381,380,374,373,390,249]]
ids_flat = ids[0]+ids[1]
ids_large = [[33,246,161,160,159,158,157,173,133,7,163,144,145,153,154,155,222,223,224,225,228,229,230,231],
      [362,398,384,385,386,387,388,466,263,382,381,380,374,373,390,249,442,443,444,445,448,449,450,451]]
ids_upper = [[221,222,223,224,225], [441,442,443,444,445]]
ids_lower = [[117,118,119,120,121], [448,449,450,451,452]]
strs = ['L','R']

from cv.landmark_detector import LandmarkDetector

'''
Image: RGB image.
Returns eye, a list of 2 - left (physically right) + right (physically left) eye images in [0,255] float32 type (if preprocess=False).
'''
def crop_eye(image, landmarks=None, pad=(0.4,0.4), large_mode=False, preprocess=False, size=None, flip_right=False,
             preserve_aspect_ratio=True):
    h = image.shape[0]
    w = image.shape[1]
    if landmarks is None:
        ld = LandmarkDetector()
        l2d_raw = ld.landmarks_2D(image)
        if l2d_raw is None: return None
    else:
        l2d_raw = landmarks
    eyes_coords = []
    eyes = []
    for i in range(2):
        l2d = l2d_raw[ids[i], :]
        xmin = np.min(l2d[:, 0])
        xmax = np.max(l2d[:, 0])
        if large_mode:
            l2d_lower = l2d_raw[ids_lower[i], :]
            l2d_upper = l2d_raw[ids_upper[i], :]
            ymin = np.mean(l2d_upper[:, 1])
            ymax = np.mean(l2d_lower[:, 1])
        else:
            ymin = np.min(l2d[:, 1])
            ymax = np.max(l2d[:, 1])
        dx = xmax-xmin
        dy = ymax-ymin
        xmin = max(0, xmin - dx * pad[0])
        ymin = max(0, ymin - dy * pad[1])
        xmax = min(w, xmax + dx * pad[0])
        ymax = min(h, ymax + dy * pad[1])
        if preserve_aspect_ratio:
            ar = h/w 
            dx = xmax - xmin
            dy = ymax - ymin
            cx = 0.5*(xmax+xmin)
            cy = 0.5*(ymax+ymin)
            if dy/dx < ar: # pad y
                extra_pad = (dx * ar - dy) // 2
                ymin -= extra_pad
                ymax += extra_pad
            elif dy/dx > ar: # pad x
                extra_pad = (dy - dx * ar) // 2
                xmin -= extra_pad
                xmax += extra_pad

        xmin = int(xmin)
        xmax = int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)

        eye = image[ymin:ymax, xmin:xmax, ...].astype("float32")
        if size:
            eye = cv2.resize(eye, size)
        if i == 1 and flip_right:
            eye = cv2.flip(eye, 1)
        if preprocess:
            eye = image_preprocess(eye)
        else:
            eye = cv2.normalize(eye, eye, 0, 255, cv2.NORM_MINMAX)
        eyes.append(eye)

        eyes_coords.append([xmin, ymin, xmax, ymax])

    return eyes_coords, eyes

def crop_upper(image, landmarks=None, size=(150,90), demo=False, pad=(0.1, 0,1), preprocess=False, **kwargs):
    h = image.shape[0]
    w = image.shape[1]
    if landmarks is None:
        ld = LandmarkDetector()
        l2d_raw = ld.landmarks_2D(image)
    else:
        l2d_raw = landmarks

    l2d = l2d_raw[ids_flat, :]
    xmin = np.min(l2d[:, 0])
    xmax = np.max(l2d[:, 0])
    ymin = np.min(l2d[:, 1])
    ymax = np.max(l2d[:, 1])
    dx = xmax - xmin
    dy = ymax - ymin
    xmin = max(0, int(xmin - dx * pad[0]))
    ymin = max(0, int(ymin - dy * pad[1]))
    xmax = min(w, int(xmax + dx * pad[0]))
    ymax = min(h, int(ymax + dy * pad[1]))
    out = image[ymin:ymax, xmin:xmax, ...].astype("float32")
    if size:
        out = cv2.resize(out, size)
    # out /= np.amax(out)
    if preprocess:
        out = image_preprocess(out, **kwargs)
    else:
        out = cv2.normalize(out, out, 0, 1, cv2.NORM_MINMAX)
    if demo:
        import matplotlib.pyplot as plt
        plt.imshow(out)
        plt.show()
    return out