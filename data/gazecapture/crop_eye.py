# this is a bit different from cv/crop_eye.py as it's based on the 14-pts model in 2d_to_3d.py instead of MediaPipe face mesh.

import numpy as np
import cv2

ids = [[9,10],
      [11,12]]
ids_flat = ids[0]+ids[1]

def crop_eye(image, landmarks, size=(150,90), demo=False, pad=0.1, flip_right=False):
    h = image.shape[0]
    w = image.shape[1]  
    eyes = []
    for i in range(2):
        l2d = landmarks[ids[i], :]
        xmin = np.min(l2d[:, 0])
        xmax = np.max(l2d[:, 0])
        ymid = np.mean(l2d[:, 1])
        # ymin = np.min(l2d[:, 1])
        # ymax = np.max(l2d[:, 1])
        dx = xmax - xmin
        xmin = max(0, int(xmin - dx * pad))
        xmax = min(w, int(xmax + dx * pad))

        dx = xmax - xmin
        dy_half = dx * 0.6 / 2
        ymin = int(ymid - dy_half)
        ymax = int(ymid + dy_half)

        eye = image[ymin:ymax, xmin:xmax, ...]
        if size:
            eye = cv2.resize(eye, size)
        if i==1 and flip_right:
            eye = cv2.flip(eye, 1)

        eyes.append(eye)
        if demo:
            import matplotlib.pyplot as plt
            plt.imshow(eye)
            plt.show()
            # plt.imshow(image)
            # plt.scatter(l2d_raw[:,0], l2d_raw[:,1])
            # plt.show()
    return eyes