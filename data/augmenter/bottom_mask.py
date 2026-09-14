'''
by Zheng.
Zhenhao's notes: works for 224x224 only.
TODO: modify the consts to relative size.
'''

import numpy as np
import cv2
from math import tan, radians
import random


def bottom_mask(rgb):
    
    # l2d = l2d.astype(int)
    
    # top_point = l2d[4, ...]

    h, w = rgb.shape()[0], rgb.shape()[1]
    top_point = [h * 0.5, w * (135 / 224)] # takes any size of input
    # top_point = [112, 135] # pre-set based on experiment

    bias = random.randint(0, 20)
    final_point = [top_point[0], top_point[1] + bias]
    
    random_ang = random.uniform(-15.0, 15.0)
    
    limit_1 = final_point[1] - final_point[0]*tan(radians(random_ang))
    limit_r = final_point[1] + (224 - final_point[0])*tan(radians(random_ang))

    points = np.array([final_point, [0, 224], [224, 224], [0, limit_1], [224, limit_r]], np.int32)
    # print(points)
    hull = cv2.convexHull(points)
    # print(hull)
    mask = cv2.fillPoly(np.zeros_like(rgb), [hull], (255,255,255))

    return mask
