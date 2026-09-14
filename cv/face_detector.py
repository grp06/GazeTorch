'''
Face detector based on Mediapipe TFLite models. REQUIRES:
cv/modules/face_detection_front.tflite
cv/modules/anchors.csv
'''

# use face_detection_front.tflite + no NMS
import csv
import cv2
import numpy as np
import tensorflow as tf
import os

from tools.utils import pad_to_square

class FaceDetector():
    def __init__(self, model_dir="cv/modules"):
        '''
        model_dir: the directory that contains face_detection_front.tflite + anchors.csv
        '''
        # inference with tflite model
        face_model_path = os.path.join(model_dir, "face_detection_front.tflite")
        anchors_path = os.path.join(model_dir, "anchors.csv")

        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=face_model_path)
        self.interpreter.allocate_tensors()

        # reading the SSD anchors
        with open(anchors_path, "r") as csv_f:
            self.anchors = np.r_[
                [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]

        # reading tflite model paramteres
        output_details = self.interpreter.get_output_details()
        input_details = self.interpreter.get_input_details()
        # print('face.output_details:', output_details)
        # print('face.input_details:', input_details)

        self.in_idx = input_details[0]['index']
        self.out_reg_idx = output_details[0]['index']
        self.out_clf_idx = output_details[1]['index']

    def __call__(self, rgb):
        # pad image to square
        img_pad, pad_half = pad_to_square(rgb)
        img_pad = img_pad.astype("float32")
        scale = max(img_pad.shape) / 128
        img_pad = cv2.resize(img_pad, (128, 128))
        img_norm = img_pad/127.5 - 1

        # run tflite model
        self.interpreter.set_tensor(self.in_idx, img_norm[None])
        self.interpreter.invoke()
        id = np.argmax(self.interpreter.get_tensor(self.out_clf_idx)[0,:,0])
        anchors = self.anchors[id]
        detect = self.interpreter.get_tensor(self.out_reg_idx)[0, id]

        #detection => bbox
        #print("Anchor:", anchors)
        #print("Raw detection result: ", detect)

        center = anchors[:2] * 128
        detect[:2] += center

        #print("Detection+centre: ", detect[:2])

        detect *= scale
        detect[:2] -= pad_half[::-1]
        #print("Detection+centre+pad: ", detect[:2])
        return detect[:4]

    # run face detection and crop the face patch.
    # returns: the cropped face image & the bounding box position
    def run_and_crop(self, rgb, ratio=1.):
        w, h, _ = rgb.shape
        bounding_box = self.__call__(rgb)
        cx = bounding_box[0]; cy = bounding_box[1]
        bw = bounding_box[2]; bh = bounding_box[3]
        x1 = int(max(cx - bw * ratio * 0.5, 0))
        x2 = int(min(cx + bw * ratio * 0.5, w))
        y1 = int(max(cy - bh * ratio * 0.5, 0))
        y2 = int(min(cy + bh * ratio * 0.5, h))
        return rgb[y1:y2, x1:x2, :], [x1, x2, y1, y2]