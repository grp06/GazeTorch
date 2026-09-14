'''
By Omid & Mahak. Head pose estimation based on a CNN model:
https://github.com/1996scarlet/Dense-Head-Pose-Estimation
Please place the tflite model (dense_face.tflite) at cv/modules before use.
This code file relies on face_detector.py so follow the notes of it too.

TF version:
dense_face.tflite passes test on TF2.3.0; it fails on TF1 and TF2.0.0

Sample usage:
frame = cv2.imread("../../0.jpg") #BGR
fd = HeadposeDetector(model_dir="../../modules-face-head-detector")
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
angles = fd(frame_rgb)
print(angles)
'''
import cv2
import numpy as np
import tensorflow as tf
import os
from functools import partial

from cv.face_detector import FaceDetector

class HeadposeDetector():
    def __init__(self, model_dir="cv/modules"):
        '''
        model_dir: the directory that contains dense_face.tflite + face_detection_front.tflite + anchors.csv
        '''
        # inference with tflite model
        headpose_model_path = os.path.join(model_dir, "dense_face.tflite")
        self.fa = DenseFaceReconstruction(headpose_model_path)
        self.model_dir = model_dir

    def __call__(self, rgb):
        # we need rgb image
        face_detector = FaceDetector(self.model_dir)
        boxes = face_detector(rgb)
        boxes_new = self._helper(boxes)

        # Now, I'll generate the headpose angles

        ##################### Main code after the face detection process #################

        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        feed = frame_bgr.copy() #BGR

        angles = (self.fa).get_landmarks(feed, boxes_new) # Order: pitch, yaw, roll

        return angles


    
    def _helper(self, value_box):
        center_x = value_box[0]
        center_y = value_box[1]
        width = value_box[2]
        height = value_box[3]


        x1 = center_x - (width/2)
        y1 = center_y - (height/2)
        x2 = center_x + (width/2)
        y2 = center_y + (height/2)
        boxes_new = np.asarray([[x1, y1, x2, y2]])
        return boxes_new
    
# Some other classes thatare important to run the current model

class BaseTFLiteFaceAlignment():
    def __init__(self, model_path, num_threads=1):
        # tflite model init
        self._interpreter = tf.lite.Interpreter(model_path=model_path)
                                                # num_threads=num_threads)
        self._interpreter.allocate_tensors()

        # model details
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        # shape details
        self._input_shape = tuple(input_details[0]["shape"][-2:])
        self._edge_size = self._input_shape[-1]
        self._trans_distance = self._edge_size / 2.0

        # inference helper
        self._set_input_tensor = partial(self._interpreter.set_tensor,
                                         input_details[0]["index"])
        self._get_camera_matrix = partial(self._interpreter.get_tensor,
                                          output_details[0]["index"])
        self._get_landmarks = partial(self._interpreter.get_tensor,
                                      output_details[1]["index"])

    def _preprocessing(self, img, bbox, factor=2.7):
        """Pre-processing of the BGR image. Adopting warp affine for face corp.

        Arguments
        ----------
        img {numpy.array} : the raw BGR image.
        bbox {numpy.array} : bounding box with format: {x1, y1, x2, y2, score}.

        Keyword Arguments
        ----------
        factor : max edge scale factor for bounding box cropping.

        Returns
        ----------
        inp : input tensor with NHWC format.
        M : warp affine matrix.
        """

        maximum_edge = max(bbox[2:4] - bbox[:2]) * factor
        scale = self._edge_size * 2.0 / maximum_edge
        center = (bbox[2:4] + bbox[:2]) / 2.0
        cx, cy = self._trans_distance - scale * center

        M = np.array([[scale, 0, cx], [0, scale, cy]])

        cropped = cv2.warpAffine(img, M, self._input_shape, borderValue=0.0)
        rgb = cropped[:, :, ::-1].astype(np.float32) #BGR=>RGB

        cv2.normalize(rgb, rgb, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)

        inp = rgb.transpose(2, 0, 1)[None]

        return inp, M

    def _inference(self, input_tensor):
        self._set_input_tensor(input_tensor)
        self._interpreter.invoke()

    def _decode_landmarks(self, iM):
        raise NotImplementedError()

    def _postprocessing(self, M):
        iM = cv2.invertAffineTransform(M)

        R = self._get_camera_matrix()[0]

        landmarks = self._decode_landmarks(iM)

        return landmarks, R

    def get_landmarks(self, image, detected_faces=None):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

        Arguments
        ----------
        image {numpy.array} : The input image. BGR.

        Keyword Arguments
        ----------
        detected_faces {list of numpy.array} : list of bounding boxes, one for each
        face found in the image (default: {None}, format: {x1, y1, x2, y2, score})
        """
        # it will be just one '
        rxyz = 0
        for box in detected_faces:
            if box[2] - box[0] < 50: # Note(2025): Original (GazePersona) = "100", later changed to "50"
                continue
            inp, M = self._preprocessing(image, box)
            self._inference(inp)

            _, R = self._postprocessing(M)
            rxyz = self.rotMatrixToEulerAngles(R)
        return rxyz # [roll , pitch , yaw] 

        
    def rotMatrixToEulerAngles(self, R):
        '''
        Ref: https://stackoverflow.com/a/15029416
        '''
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        if sy < 1e-6:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        else:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])

        return np.degrees([x, y, z])


class DenseFaceReconstruction(BaseTFLiteFaceAlignment):
    def _decode_landmarks(self, iM):
        points = self._get_landmarks()[0]

        points *= iM[0][0]
        points[:, :2] += iM[:, -1]

        return points