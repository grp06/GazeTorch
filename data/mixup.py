'''
Util functions for mixup
'''

import cv2
import numpy as np

def calculateIOU(img1_left_eye, img1_right_eye, img2_left_eye, img2_right_eye):
    # By Mahak. All input images should be binary.
    d1 = sum(sum(img1_left_eye == 1) + sum(img1_right_eye == 1))
    d2 = sum(sum(img2_left_eye == 1) + sum(img2_right_eye == 1))

    n1_img = img1_left_eye + img2_left_eye
    n2_img = img1_right_eye + img2_right_eye

    n = sum(sum(n1_img == 2) + sum(n2_img == 2))
    d = d1 + d2 - n 

    return n/d

class MixupProcessor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def mix_images(img1, img2, alpha):
        '''
        Mixup two images.
        img1/2:
            * `np.ndarray`: the image (recommended to be uint8)
            * `str`: path to the image
        alpha: img1 weight
        '''
        if isinstance(img1, str):
            img1 = cv2.imread(img1)
        if isinstance(img2, str):
            img2 = cv2.imread(img2)
        # blend_images
        beta = (1.0 - alpha)
        dst = cv2.addWeighted(img1, alpha, img2, beta, 0.0)

        return dst
    
    @staticmethod
    def eye_iou(segmap1, segmap2):
        '''
        Get IOU between two eye maps from segmap(v3).
        segmap1/2:
            * `np.ndarray`: the image (recommended to be uint8)
            * `str`: path to the image
        '''
        if isinstance(segmap1, str):
            segmap1 = cv2.imread(segmap1, cv2.IMREAD_UNCHANGED)
        if isinstance(segmap2, str):
            segmap2 = cv2.imread(segmap2, cv2.IMREAD_UNCHANGED)

        # binary map
        img1_left_eye = segmap1[:, :, 0]
        img1_left_eye[img1_left_eye == 2] = 0
        img1_right_eye = segmap1[:, :, 2]
        img1_right_eye[img1_right_eye == 2] = 0

        img2_left_eye = segmap2[:, :, 0]
        img2_left_eye[img2_left_eye == 2] = 0
        img2_right_eye = segmap2[:, :, 2]
        img2_right_eye[img2_right_eye == 2] = 0

        return calculateIOU(img1_left_eye, img1_right_eye,
                            img2_left_eye, img2_right_eye)