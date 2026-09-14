import cv2
import numpy as np
import random
from tools.utils import rotate_cv2, rotate_vector, xyz_to_pitchyaw, pitchyaw_to_xyz

'''
T2D: for rotation, angle = counter-clockwise
'''

class Transform2D:
    def __init__(self, size=224, buffer_size_ratio=2, 
        scale_min=0.8, scale_max=1.2, trans_max=0.1, rotate_max=0):
        self.size = size
        self.buffer_size = size * buffer_size_ratio
        self.buffer = np.zeros((self.buffer_size, self.buffer_size, 4))
        self.c = self.buffer_size // 2

        self.scale_min = scale_min
        self.scale_max = scale_max
        self.trans_max = trans_max
        self.rotate_max = rotate_max

        self.buffer_angle = 0

    def __call__(self, image, params=None, is_rgb=True):
        self.buffer.fill(0)
        ndims = image.shape[2] if image.ndim==3 else 1

        if params is None:
            params = self.generate_params()

        interpolation = cv2.INTER_LINEAR if is_rgb else cv2.INTER_NEAREST

        wr = params['w']
        hr = params['h']
        transX = params['tX']
        transY = params['tY']
        rotA = params['r']

        x1 = self.c - wr // 2 + transX
        y1 = self.c - hr // 2 + transY

        rgb_resized = cv2.resize(image, (wr, hr), interpolation=interpolation)
        
        if rotA != 0:
            rgb_resized = rotate_cv2(rgb_resized, rotA, interpolation=interpolation)
            hr = rgb_resized.shape[0]
            wr = rgb_resized.shape[1]
            self.buffer_angle = rotA
        
        self.buffer[y1:y1+hr, x1:x1+wr, :ndims] = rgb_resized

        result = self.buffer[self.c-self.size//2:self.c+self.size//2,
                 self.c-self.size//2:self.c+self.size//2, :ndims].copy()
        return result

    def generate_params(self):
        '''
        Generate a random set of transform params.
        Return a dict storing them.
        '''
        ratioX = self.random(self.scale_min, self.scale_max)
        ratioY = ratioX * self.random(self.scale_min, self.scale_max)
        wr = int(self.size * ratioX)
        hr = int(self.size * ratioY)
        transX = int(self.random(-self.trans_max, self.trans_max) * self.size) if self.trans_max>0 else 0
        transY = int(self.random(-self.trans_max, self.trans_max) * self.size) if self.trans_max>0 else 0
        rotA = self.random(-self.rotate_max, self.rotate_max) if self.rotate_max>0 else 0

        params = {
            'w': wr,
            'h': hr,
            'tX': transX,
            'tY': transY,
            'r': rotA
        }
        self.buffer_params = params
        return params

    @staticmethod
    def random(v_min, v_max):
        return random.random() * (v_max - v_min) + v_min

def rotate_pitchyaw(g, angle):
    g_xyz = pitchyaw_to_xyz(g)
    g_xyz = rotate_vector(g_xyz, -angle)
    return xyz_to_pitchyaw(g_xyz)