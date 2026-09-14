'''
Landmark detector interface.
Class LandmarkDetector supports 4 underlying models: dlib, fa, mp and mp_tflite.
mp (mediapipe) is easiest to deploy - just make sure mediapipe library is installed.
For other 3 models, please put corresponding model files at "./cv/modules" in prior.
'''

import pickle
import numpy as np
import os

class LandmarkDetector:
    def __init__(self, method='mp', model_dir='cv/modules'):
        '''
        model_dir: the directory that contains related TFLITE models
        * dlib: shape_predictor_68_face_landmarks.dat
        * fa: FA2.pkl, FA3.pkl
        * mp: nothing needed
        * mp_tflite: face_landmark.tflite. ONLY WORKS FOR BOUNDING BOXES given by FaceDetector. ONLY WORKS with TF version>=1.15.2
        '''
        self.method = method
        if method=='dlib':
            import dlib
            self._2d = dlib.shape_predictor(os.path.join(model_dir, "shape_predictor_68_face_landmarks.dat"))
            with open(r'modules/FA3.pkl', 'rb') as f:
                self._3d = pickle.load(f)
            self.len = 68
        elif method=='fa':
            with open(os.path.join(model_dir, 'modules/FA2.pkl', 'rb')) as f:
                self._2d = pickle.load(f)
            with open(os.path.join(model_dir, 'modules/FA3.pkl', 'rb')) as f:
                self._3d = pickle.load(f)
            self.len = 68
        elif method=='mp':
            import mediapipe as mp
            mp_face_mesh =  mp.solutions.face_mesh
            self._3d = mp_face_mesh.FaceMesh(
                # static_rgb_mode=True, # LOG(Jun 10 2022): MP on server doesn't take this arg
                max_num_faces=1,
                min_detection_confidence=0.5)
            self.len = 468
        elif method=='mp_tflite': # only work for bounding boxes
            import tensorflow as tf
            from cv.face_detector import FaceDetector
            model_path = os.path.join(model_dir, "face_landmark.tflite")
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            self._3d = interpreter
            self._fd = FaceDetector()
            self.len = 468
        else:
            raise ValueError("[Gaze]Invalid landmark detection method.")

        # landmark IDs
        if not method.startswith('mp'):
            # id_calib = [36, 33, 8, 45, 48, 54] # Rasoul's Pnp code
            # ID_CALIB = [36, 39, 42, 45, 31, 35, 30, 8]
            self.ID_CALIB = [36, 39, 42, 45, 31, 35]
            self.ID_NOSETIP = 30
            self.ID_EYE_INNER = [39, 42]
            self.ID_EYE_OUTER = [36, 45]
        else:
            # ID_CALIB = [33,133,362,263,159,145,386,374,48,4,278]
            self.ID_CALIB = [33, 133, 362, 263, 219, 439]
            self.ID_NOSETIP = 4
            self.ID_EYE_INNER = [133, 362]
            self.ID_EYE_OUTER = [33, 263]

    def landmarks_3D(self, rgb, tflite_face=True):
        if self.method in ['fa','dlib']:
            return self._3d.get_landmarks(rgb)[0]
        elif self.method=='mp':
            result = self._3d.process(rgb).multi_face_landmarks
            if result is None: return None
            l3d_raw = mp_landmarklist_to_array(
                result[0]
            ) # a "NormalizedLandmarkList" object
            l3d_raw[:,1] *= rgb.shape[0]/rgb.shape[1]
            return l3d_raw
        elif self.method=='mp_tflite':
            if tflite_face:
                h = rgb.shape[0]
                w = rgb.shape[1]
                bbox = self._fd(rgb)
                scale = 1.5
                x1 = int(max(0, bbox[0] - bbox[-2]*0.5*scale))
                x2 = int(min(w, bbox[0] + bbox[-2]*0.5*scale))
                y1 = int(max(0, bbox[1] - bbox[-1]*0.5*scale))
                y2 = int(min(h, bbox[1] + bbox[-1]*0.5*scale))
                rgb = rgb[y1:y2, x1:x2, ...]
            hf = rgb.shape[0]
            wf = rgb.shape[1]

            if np.amax(rgb)>1:
                rgb = rgb.astype("float32")
            rgb /= np.amax(rgb)
            from cv2 import resize
            rgb = resize(rgb, (192, 192))
            rgb = rgb[np.newaxis, ...]
            self._3d.set_tensor(0, rgb)
            self._3d.invoke()
            l3d_raw = self._3d.get_tensor(213).reshape((468,3))
            l3d_raw[:,0] *= wf/192
            l3d_raw[:,1] *= hf/192
            l3d_raw[:, 2] *= wf/ 192
            return l3d_raw

    def landmarks_2D(self, rgb, head_rect=None, tflite_face=True):
        # head_rect is used in dlib mode only
        if self.method=='dlib':
            landmarks = self._2d(rgb, head_rect)
            landmarks = dlib_shape_to_np(landmarks)
            return landmarks
        elif self.method=='fa':
            return self._2d.get_landmarks(rgb)[0]
        elif self.method=='mp':
            # NOTES: MULTIPIPE OUTPUTS [0,1] LANDMARKS! CONVERT THEM INTO PIXEL ID BEFORE USE
            result = self._3d.process(rgb).multi_face_landmarks
            if result is None: return None
            l2d_raw = mp_landmarklist_to_array(
                result[0], to_3d=False
            )
            l2d_raw[:,0] *= rgb.shape[1]
            l2d_raw[:,1] *= rgb.shape[0]
            return l2d_raw
        elif self.method == 'mp_tflite':
            if tflite_face:
                h = rgb.shape[0]
                w = rgb.shape[1]
                bbox = self._fd(rgb)
                scale = 1.5
                x1 = int(max(0, bbox[0] - bbox[-2] * 0.5 * scale))
                x2 = int(min(w, bbox[0] + bbox[-2] * 0.5 * scale))
                y1 = int(max(0, bbox[1] - bbox[-1] * 0.5 * scale))
                y2 = int(min(h, bbox[1] + bbox[-1] * 0.5 * scale))
                rgb = rgb[y1:y2, x1:x2, ...]
            hf = rgb.shape[0]
            wf = rgb.shape[1]

            if np.amax(rgb) > 1:
                rgb = rgb.astype("float32")
            rgb /= np.amax(rgb)
            from cv2 import resize
            rgb = resize(rgb, (192, 192))
            rgb = rgb[np.newaxis, ...]
            self._3d.set_tensor(0, rgb)
            self._3d.invoke()
            l3d_raw = self._3d.get_tensor(213).reshape((468, 3))
            l2d_raw = l3d_raw[:, :2]
            l2d_raw[:, 0] *= wf / 192
            l2d_raw[:, 0] += x1
            l2d_raw[:, 1] *= hf / 192
            l2d_raw[:, 1] += y1
            return l2d_raw.copy()

def mp_landmarklist_to_array(landmark_list, to_3d=True):
    n = 468
    d = 3 if to_3d else 2
    ans = np.zeros((n,d))
    for i in range(n):
        ans[i,0] = landmark_list.landmark[i].x
        ans[i,1] = landmark_list.landmark[i].y
        if to_3d:
            ans[i,2] = landmark_list.landmark[i].z
    # Nov 30 update: reverse X, Y, Z
    return ans

def dlib_shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords