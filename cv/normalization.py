# Data normalization (ETRA'18) based on 3D landmarks + camera matrix

import numpy as np
import cv2

# Pipeline: image => normalize => warped_image => gaze(norm) => gaze = inv(M)*gaze(norm)
def normalize(img, cam, face_model, rvec, tvec, id_calib=(36, 39, 42, 45, 31, 35), roiSize=(224,224)):
    ## normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600 * (224 / roiSize[0])  # normalized distance between eye and camera
    # roiSize = (224, 224)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = tvec.reshape((3, 1))
    hR, _ = cv2.Rodrigues(rvec) # rotation matrix
    #
    # pose_mat = cv2.hconcat((hR, np.array([[0.], [0], [0]])))
    # _, _, _, _, _, _, pyr = cv2.decomposeProjectionMatrix(pose_mat)

    if id_calib:
        face_model = face_model[id_calib, :]
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    #
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # two_eye_center = np.mean(face_model[id_calib[:4], :].T, axis=1).reshape((3, 1))
    # nose_center = np.mean(face_model[id_calib[4:], :].T, axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    ## ---------- normalize image ----------
    distance = np.linalg.norm(face_center)  # actual distance between eye and original camera
    # distance = np.linalg.norm(tvec)

    z_scale = distance_norm / distance
    cam_norm = np.array([  # camera intrinsic parameters of the virtual camera
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    # forward = (tvec / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    # M = np.dot(S, R)

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)  # warp the input image

    # head pose after normalization
    # hR_norm = np.dot(R, hR)  # head pose rotation matrix in normalized space
    # hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    # # normalize the facial landmarks
    # num_point = landmarks.shape[0]
    # landmarks_warped = cv2.perspectiveTransform(landmarks, W)
    # landmarks_warped = landmarks_warped.reshape(num_point, 2)

    return img_warped, R, W
