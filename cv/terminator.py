import cv2
import numpy as np
from cv.crop_eye import crop_eye
import matplotlib.pyplot as plt
from cv.iris_detector import MPIris
from cv.iris_detector import iris_cdf
from cv.face_detector import FaceDetector
from cv.landmark_detector import LandmarkDetector

"""
by Mehdi.
Terminator class returns all the coordinates and 
eye patch sizes in the original size of the image and coordinates
"""

# Key constants
# face (given any image)
FACE_COORD = "face_coord"
FACE_PATCH = "face_patch"
# eye patches
EYE_L = "leye_patch"
EYE_R = "reye_patch"
IRISMAP_L = "liris_mask"
IRISMAP_R = "riris_mask"
POLY_L = "poly_leye"
POLY_R = "poly_reye"
# result
POS_L = "leye_coord"
POS_R = "reye_coord"
MP_IRIS_L = "mp_liris_landmarks"
MP_IRIS_R = "mp_riris_landmarks"
MP_EYE_L = "leye_landmarks"
MP_EYE_R = "reye_landmarks"
MASKS_STACKED = "stacked_face_masks"

class Terminator:
    def __init__(self, iris_model_dir, face_detector_module_path=None):
        self.face_detector_module_path = face_detector_module_path
        self.iris_model_dir = iris_model_dir
        self.mp = MPIris(self.iris_model_dir)
        self.ld = LandmarkDetector()
        self.result = {FACE_COORD:None, MASKS_STACKED:None, IRISMAP_L:None, IRISMAP_R:None, POLY_L: None, POS_R:None,
         POS_L:None, POS_R:None, MP_IRIS_L: None, MP_IRIS_R:None, FACE_PATCH:None,
          EYE_L:None, EYE_R:None, MP_EYE_L: None, MP_EYE_R: None}
        
    def run(self, rgb, size=None, demo=False, skip_left=False, skip_right=False):
        self.rgb = rgb
        self.result = None
        self.result = {FACE_COORD:None, MASKS_STACKED:None, IRISMAP_L:None, IRISMAP_R:None, POLY_L: None, POS_R:None,
         POS_L:None, POS_R:None, MP_IRIS_L: None, MP_IRIS_R:None, FACE_PATCH:None,
          EYE_L:None, EYE_R:None, MP_EYE_L: None, MP_EYE_R: None}
        if self.face_detector_module_path:
            self.compute_face_coord(use_face_detector=True)
        else:
            self.compute_face_coord(use_face_detector=False)
        if not self.compute_eye_coord():
            print("[Terminator]Failed to locate eyes.")
            return None
        self.result[FACE_PATCH] = self.rgb[self.result[FACE_COORD][1]: self.result[FACE_COORD][3], self.result[FACE_COORD][0]: self.result[FACE_COORD][2]]
        self.compute_masks_landmarks(threshold_mask_eye=None, threshold_mask_iris=None)
        self.get_stacked_face_masks(size=size, skip_left=skip_left, skip_right=skip_right)
        if demo:
            return self.visualize(demo=demo)
        else:
            return self.result[MASKS_STACKED]
    
    def compute_face_coord(self, use_face_detector=False):
        if use_face_detector:
            fd = FaceDetector(self.face_detector_module_path)
            face_bbox_scaled = fd(self.rgb)
            scale = 1.5
            h = self.rgb.shape[0]
            w = self.rgb.shape[1]
            x1 = int(max(0, face_bbox_scaled[0] - face_bbox_scaled[-2]*0.5*scale))
            x2 = int(min(w, face_bbox_scaled[0] + face_bbox_scaled[-2]*0.5*scale))
            y1 = int(max(0, face_bbox_scaled[1] - face_bbox_scaled[-1]*0.5*scale))
            y2 = int(min(h, face_bbox_scaled[1] + face_bbox_scaled[-1]*0.5*scale))
            self.result[FACE_COORD] = (x1, y1, x2, y2)
        else:
            self.result[FACE_COORD] = (0, 0, self.rgb.shape[1], self.rgb.shape[0])

    def compute_eye_coord(self):
        if self.result[FACE_COORD] is None:
            raise ValueError('Face coordinates are None! call compute_face_coord first!')
        face_coord = self.result[FACE_COORD]
        # relative_eyes_coord, eyes = crop_eye(self.rgb[face_coord[1]: face_coord[3], face_coord[0]: face_coord[2], :])
        landmarks = self.ld.landmarks_2D(self.rgb)
        eyes_result = crop_eye(self.rgb, landmarks)
        if eyes_result is None: return False
        relative_eyes_coord, eyes = eyes_result
        shift_x, shift_y = face_coord[0], face_coord[1]
        eyes_coord = []
        for eye_coord in relative_eyes_coord:
            eyes_coord.append([eye_coord[0]+shift_x, eye_coord[1]+shift_y, eye_coord[2]+ shift_x, eye_coord[3]+shift_y])

        self.result[POS_L] = eyes_coord[0]
        self.result[POS_R] = eyes_coord[1]
        self.result[EYE_L] = eyes[0]
        self.result[EYE_R] = eyes[1]
        if eyes[0] is None or eyes[1] is None:
            return False

        return True
    
    
    def compute_masks_landmarks(self, threshold_mask_eye=None, threshold_mask_iris=None):
        if (self.result[POS_L] is None) or (self.result[POS_R] is None):
            raise ValueError('Eye coordinates are None! call compute_eye_coord first!')
        

        # resize left and right eye to 150*90
        scale_ly = 90 / self.result[EYE_L].shape[0]
        scale_lx = 150 / self.result[EYE_L].shape[1]
        resized_leye = cv2.resize(self.result[EYE_L], None, fx=scale_lx, fy=scale_ly)
        # same for right eye
        scale_ry = 90 / self.result[EYE_R].shape[0]
        scale_rx = 150 / self.result[EYE_R].shape[1]
        resized_reye = cv2.resize(self.result[EYE_R], None, fx=scale_rx, fy=scale_ry)

        relative_resized_leye_landmarks, mp_relative_liris_landmarks = self.mp(resized_leye)
        relative_flipped_resized_reye_landmarks, mp_relative_riris_landmarks = self.mp(resized_reye[:, ::-1]) # reye should be flipped
        # flip the coordinate again
        relative_resized_reye_landmarks = [(resized_reye.shape[1]-x, y) for (x, y) in relative_flipped_resized_reye_landmarks]
        relative_resized_reye_landmarks = np.array(relative_resized_reye_landmarks)
        self.result[MP_EYE_L] = np.array([(1/scale_lx * x+self.result[POS_L][0], 1/scale_ly * y+self.result[POS_L][1]) for (x, y) in relative_resized_leye_landmarks])
        self.result[MP_EYE_R] = np.array([(1/scale_rx * x+self.result[POS_R][0], 1/scale_ry * y+self.result[POS_R][1]) for (x, y) in relative_resized_reye_landmarks])

        mp_relative_riris_landmarks = [(resized_reye.shape[1]-x, y) for (x, y) in mp_relative_riris_landmarks]
        mp_relative_riris_landmarks = np.array(mp_relative_riris_landmarks)
        
        self.result[MP_IRIS_R] = np.array([(1/scale_rx * x+self.result[POS_R][0], 1/scale_ry * y+self.result[POS_R][1]) for (x, y) in mp_relative_liris_landmarks])
        self.result[MP_IRIS_L] = np.array([(1/scale_lx * x+self.result[POS_L][0], 1/scale_ly * y+self.result[POS_L][1]) for (x, y) in mp_relative_riris_landmarks])

        _, liris_mask, _, _, poly_leye = iris_cdf(resized_leye, relative_resized_leye_landmarks)
        _, riris_mask, _, _, poly_reye = iris_cdf(resized_reye, relative_resized_reye_landmarks)
 
        riris_mask = cv2.resize(riris_mask, None, fx=1/scale_rx, fy=1/scale_ry)
        poly_reye = cv2.resize(poly_reye, None, fx=1/scale_rx, fy=1/scale_ry)

        liris_mask = cv2.resize(liris_mask, None, fx=1/scale_lx, fy=1/scale_ly)
        poly_leye = cv2.resize(poly_leye, None, fx=1/scale_lx, fy=1/scale_ly)

        if threshold_mask_eye is not None:
            poly_leye = (poly_leye > threshold_mask_eye)
            poly_reye = (poly_reye > threshold_mask_eye)
        if threshold_mask_iris is not None:
            riris_mask = (riris_mask > threshold_mask_iris)
            liris_mask = (liris_mask > threshold_mask_iris)

        self.result[IRISMAP_L] = liris_mask
        self.result[IRISMAP_R] = riris_mask
        self.result[POLY_R] = poly_reye
        self.result[POLY_L] = poly_leye

    def get_stacked_face_masks(self, size=None, skip_left=False, skip_right=False):
        face_coord = self.result[FACE_COORD]
        mask_riris = np.zeros(shape=(self.rgb.shape[0], self.rgb.shape[1]))
        mask_liris = np.zeros(shape=(self.rgb.shape[0], self.rgb.shape[1]))
        mask_reye = np.zeros(shape=(self.rgb.shape[0], self.rgb.shape[1]))
        mask_leye = np.zeros(shape=(self.rgb.shape[0], self.rgb.shape[1]))

        if not skip_left:
            mask_liris[self.result[POS_L][1]:self.result[POS_L][1]+self.result[EYE_L].shape[0], self.result[POS_L][0]:self.result[POS_L][0]+self.result[EYE_L].shape[1]] = self.result[IRISMAP_L]
            mask_leye[self.result[POS_L][1]:self.result[POS_L][1]+self.result[EYE_L].shape[0], self.result[POS_L][0]:self.result[POS_L][0]+self.result[EYE_L].shape[1]] = self.result[POLY_L]

        if not skip_right:
            mask_riris[self.result[POS_R][1]:self.result[POS_R][1]+self.result[EYE_R].shape[0], self.result[POS_R][0]:self.result[POS_R][0]+self.result[EYE_R].shape[1]] = self.result[IRISMAP_R]
            mask_reye[self.result[POS_R][1]:self.result[POS_R][1]+self.result[EYE_R].shape[0], self.result[POS_R][0]:self.result[POS_R][0]+self.result[EYE_R].shape[1]] = self.result[POLY_R]

        # crop and resize all the masks based on face coord
        mask_riris = mask_riris[face_coord[1]: face_coord[3], face_coord[0]: face_coord[2]]
        mask_liris = mask_liris[face_coord[1]: face_coord[3], face_coord[0]: face_coord[2]]
        mask_leye = mask_leye[face_coord[1]: face_coord[3], face_coord[0]: face_coord[2]]
        mask_reye = mask_reye[face_coord[1]: face_coord[3], face_coord[0]: face_coord[2]]
        face = self.rgb[face_coord[1]: face_coord[3], face_coord[0]: face_coord[2]]

        if size:
            mask_riris = cv2.resize(mask_riris, size)
            mask_liris = cv2.resize(mask_liris, size)
            mask_leye = cv2.resize(mask_leye, size)
            mask_reye = cv2.resize(mask_reye, size)
            face = cv2.resize(face, size)


        self.result[MASKS_STACKED] = np.zeros(shape=(face.shape[0], face.shape[1], 4), dtype='uint8')
        # self.result[MASKS_STACKED][:, :, 0:3] = face
        self.result[MASKS_STACKED][:, :, 0] = mask_leye.astype('uint8')
        self.result[MASKS_STACKED][:, :, 1] = mask_liris.astype('uint8')
        self.result[MASKS_STACKED][:, :, 2] = mask_reye.astype('uint8')
        self.result[MASKS_STACKED][:, :, 3] = mask_riris.astype('uint8')


    def visualize(self, demo=False):
        if self.result[MASKS_STACKED] is None: return None
        img = np.copy(self.rgb)[:, :, ::-1]
        img = img.astype('uint8')

        for l in range(16):
            cv2.circle(img, (int(self.result[MP_EYE_L][l, 0]), int(self.result[MP_EYE_L][l, 1])), 2, (0, 255, 0), -1)
            cv2.circle(img, (int(self.result[MP_EYE_R][l, 0]), int(self.result[MP_EYE_R][l, 1])), 2, (0, 255, 0), -1)
        for l in range(5):
            cv2.circle(img, (int(self.result[MP_IRIS_L][l, 0]), int(self.result[MP_IRIS_L][l, 1])), 2, (0, 0, 255), -1)
            cv2.circle(img, (int(self.result[MP_IRIS_R][l, 0]), int(self.result[MP_IRIS_R][l, 1])), 2, (0, 0, 255), -1)

        if demo:
            vis_r = self.result[POLY_R]
            vis_l = self.result[POLY_L] 
            vis_ri = self.result[IRISMAP_R]
            vis_li = self.result[IRISMAP_L]

            plt.subplot(231)
            plt.imshow(img[:, :, ::-1]/255)
            plt.subplot(232)
            plt.imshow(vis_r, cmap='gray')
            plt.subplot(233)
            plt.imshow(vis_ri, cmap='gray')
            plt.subplot(234)
            plt.imshow(vis_l, cmap='gray')
            plt.subplot(235)
            plt.imshow(vis_li, cmap='gray')
            plt.savefig('result1.png')
            plt.close()

        # visualize overlayed stacked face and masks (The "Terminator")

        face = self.face
        mask_leye = self.result[MASKS_STACKED][:, :, 0] 
        mask_liris = self.result[MASKS_STACKED][:, :, 1] 
        mask_reye = self.result[MASKS_STACKED][:, :, 2] 
        mask_riris = self.result[MASKS_STACKED][:, :, 3]


        vis_res_iris = np.zeros(shape=(face.shape[0], face.shape[1], 3), dtype='uint8')
        overlayed_iris_map = mask_liris + mask_riris
        vis_res_iris[:, :, 2] = overlayed_iris_map

        vis_res_eye = np.zeros(shape=(face.shape[0], face.shape[1], 3), dtype='uint8')
        overlayed_eye_map = mask_leye + mask_reye
        vis_res_eye[:, :, 1] = overlayed_eye_map 
        
        gray_img = cv2.cvtColor(face[:, :, ::-1], cv2.COLOR_BGR2GRAY)
        vis_img = cv2.merge([gray_img, gray_img, gray_img])
            
        overlayed_image = cv2.addWeighted(vis_img, 0.5, vis_res_eye, 0.5,0)
        overlayed_image = cv2.addWeighted(overlayed_image, 0.5, vis_res_iris, 0.5,0)

        if demo:
            plt.subplot(231)
            plt.imshow(overlayed_image[:, :, ::-1]/255)
            plt.subplot(232)
            plt.imshow(mask_riris, cmap='gray')
            plt.subplot(233)
            plt.imshow(mask_reye, cmap='gray')
            plt.subplot(234)
            plt.imshow(mask_liris, cmap='gray')
            plt.subplot(235)
            plt.imshow(mask_leye, cmap='gray')
            plt.savefig('result_stacked.png')
            plt.close()
                        
        return overlayed_image


    
if __name__ == "__main__":
    d = Terminator(r"C:\Users\s50024230\Desktop\iris_detection", r"C:\Users\s50024230\Desktop\iris_detection")
    img = cv2.imread(r"C:\Users\s50024230\AppData\Roaming\eSpace_Desktop\UserData\s50024230\ReceiveFile\mpiiface_224\8.jpg")
    out = d.run(img[:, :, ::-1], demo=True)
