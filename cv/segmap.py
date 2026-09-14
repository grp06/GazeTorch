'''
the "V3" segmentation map. with 4 channels:
    0. left eye
    1. left iris
    2. right eye
    3. right iris
for eye-channels, pixels take 0(background)/1(eye)/2(eyelid);
for iris-channels, pixels take 0(background)/1(iris).
CAUTION: NO POST_PROCESSING AFTER IMREAD!
---
facemapV2 vs segmapV3:
facemap: values are binary (0/255).
segmap: values are 0/1/2(eye-only).
'''

import cv2
import numpy as np
import joblib

import cv.consts_mediapipe as mp
from cv.landmark_detector import LandmarkDetector
from tools.utils import mix
from cv.terminator import Terminator

class Segmap:
    def __init__(self, size=224):
        self.ld = LandmarkDetector()
        self.T = Terminator("cv/modules")
        self.size = size

        # Mehdi's confidence value regressor
        self.confidence_regressor = joblib.load('cv/modules/confidence_regressor.sav')
        self.confidence_scaler = joblib.load('cv/modules/confidence_scaler.sav')

    def get_eyelid(self, rgb, demo_with_original=False):
        l2d = self.ld.landmarks_2D(rgb)
        if l2d is None: return None
        l2ds_L = l2d[mp.EYELID.LEFT_CONTOUR, ...].astype(int)
        l2ds_R = l2d[mp.EYELID.RIGHT_CONTOUR, ...].astype(int)

        mask = np.zeros_like(rgb, dtype=float)

        cv2.fillPoly(mask, [l2ds_L], [255, 0, 0])
        cv2.fillPoly(mask, [l2ds_R], [0, 255, 0])

        if demo_with_original:
            return mix(rgb.astype(float)/255, mask/255, 0.25) * 255

        return mask

    def get_eye_plus_eyelid(self, rgb):
        l2d = self.ld.landmarks_2D(rgb)
        if l2d is None: return None
        l2d_eye_L = l2d[mp.EYE.LEFT.CONTOUR, ...].astype(int)
        l2d_eye_R = l2d[mp.EYE.RIGHT.CONTOUR, ...].astype(int)
        l2d_eyelid_L = l2d[mp.EYELID.LEFT_CONTOUR, ...].astype(int)
        l2d_eyelid_R = l2d[mp.EYELID.RIGHT_CONTOUR, ...].astype(int)

        mask = np.zeros((self.size, self.size, 4), dtype=float)

        cv2.fillPoly(mask, [l2d_eye_L], [255, 0, 0, 0])
        cv2.fillPoly(mask, [l2d_eye_R], [0, 255, 0, 0])
        cv2.fillPoly(mask, [l2d_eyelid_L], [0, 0, 255, 0])
        cv2.fillPoly(mask, [l2d_eyelid_R], [0, 0, 0, 255])

        return mask

    def __call__(self, rgb):
        out_eye = self.get_eye_plus_eyelid(rgb)
        out_term = self.T.run(rgb)

        if out_eye is None or out_term is None:
            return None
        
        out = np.zeros_like(out_eye, dtype='uint8') # 4 channels
        # left eye
        out[..., 0] = out_eye[..., 0] / 255 + out_eye[..., 2] / 255 * 2
        # right eye
        out[..., 2] = out_eye[..., 1] / 255 + out_eye[..., 3] / 255 * 2
        # left iris
        out[..., 1] = out_term[..., 1] / 255
        # right iris
        out[..., 3] = out_term[..., 3] / 255

        out = np.clip(out, 0, 2)
        return out # the "segmap" v3
    
    @staticmethod
    def extract_feature(face, segmap, headpose):
        segmap = (segmap==1).astype(int)*255
        leye = segmap[:, :, 0]
        liris = segmap[:, :, 1]
        reye = segmap[:, :, 2]
        riris = segmap[:, :, 3]
        pitch = headpose[0] * 180 / np.pi
        yaw = headpose[1] * 180 / np.pi
        
        leye_pix = np.count_nonzero(leye == 255)
        reye_pix = np.count_nonzero(reye == 255)
        liris_pix = np.count_nonzero(liris == 255)
        riris_pix = np.count_nonzero(riris == 255)
        mean_illum = np.mean(face)
        feature = [leye_pix, liris_pix, reye_pix, riris_pix] + [mean_illum] + [pitch, yaw]

        return feature

    def get_confidence(self, face, segmap, headpose):
        feature = self.extract_feature(face, segmap, headpose)
        feature = np.array(feature).reshape([1, -1])
        feature_n = self.confidence_scaler.transform(feature)
        confs = self.confidence_regressor.predict(feature_n)[0]
        confs[confs < 0.25] = 0
        confs[confs >= 0.9] = 1
        confs[(np.where((confs >= 0.25) & (confs <= 0.9)))] = 1.54 * (confs[(np.where((confs >= 0.25) & (confs <= 0.9)))] - 0.25)
        confs[confs >= 1] = 1
        confs[confs <= 0] = 0
        
        return confs # channels: 0 = wE_L, 1 = wI_L, 2 = wE_R, 3 = wI_R. same as segmap/segmaps.

    @staticmethod
    # generate a demo image given a face iamge and its segmap.
    def demo(face, segmap, segmap_opacity=0.5):
        mask = np.zeros_like(face)
        mask_eye = (segmap[..., 0]==1).astype(float) + (segmap[..., 2]==1).astype(float)
        mask_eyelid = (segmap[..., 0]==2).astype(float) + (segmap[..., 2]==2).astype(float)
        mask_iris = (segmap[..., 1]!=0).astype(float) + (segmap[..., 3]!=0).astype(float)
        mask[..., 2] = mask_eye * 255
        mask[..., 1] = mask_iris * 255
        mask[..., 0] = mask_eyelid * 255
        out = mix(face.astype(float) / 255, mask / 255, segmap_opacity) * 255
        return out

    @staticmethod
    # visualize a 4-channel processed segmap(v3).
    def demo_png(rgb4):
        rgb = np.zeros((rgb4.shape[0], rgb4.shape[1], 3))
        rgb[..., 0] += (rgb4[...,0] == 2).astype(int) * 255
        rgb[..., 0] += (rgb4[...,2] == 2).astype(int) * 255
        rgb[..., 2] += (rgb4[...,0] == 1).astype(int) * 255
        rgb[..., 2] += (rgb4[...,2] == 1).astype(int) * 255
        rgb[..., 1] += (rgb4[...,1] == 1).astype(int) * 255
        rgb[..., 1] += (rgb4[...,3] == 1).astype(int) * 255
        rgb = np.clip(rgb, 0, 255)
        return rgb