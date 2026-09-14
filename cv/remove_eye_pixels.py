import cv2
import numpy as np

import cv.consts_mediapipe as const_mp

def remove_eye_pixels(rgb_face):
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

    # Detect facial landmarks
    results = face_mesh.process(rgb_face)

    if results is None: return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    h, w = rgb_face.shape[:2]

    eye_points_left = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in const_mp.EYE.LEFT.CONTOUR])
    eye_points_right = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in const_mp.EYE.RIGHT.CONTOUR])

    # Create a mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [eye_points_left], 1)
    cv2.fillPoly(mask, [eye_points_right], 1)

    mask = np.stack([mask]*3, axis=-1)

    return mask