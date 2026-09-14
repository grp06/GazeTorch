import numpy as np
import cv2
import imutils
from scipy.spatial.transform import Rotation

def find_similarity_transform(source_points, target_points, glasses_scale):
    """ This One works
    Find similarity transformation (rotation + scale + translation) between two sets of 2D points.
    
    Parameters:
    source_points: numpy array of shape (N, 2) - source landmarks
    target_points: numpy array of shape (N, 2) - target landmarks
    
    Returns:
    R: 2x2 rotation matrix
    s: scale factor
    t: 2x1 translation vector
    """
    source_points = np.array(source_points)
    target_points = np.array(target_points)
    
    # Center the points (subtract centroids)
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_centroid
    target_centered = (target_points - target_centroid)*glasses_scale
    
    # Calculate scale factor
    source_scale = np.sqrt(np.sum(source_centered**2))
    target_scale = np.sqrt(np.sum(target_centered**2))
    scale = target_scale / source_scale
    
    # Normalize the centered points
    source_normalized = source_centered / source_scale
    target_normalized = target_centered / target_scale
    
    # Calculate rotation using SVD
    # We want to find R such that target_normalized ≈ R * source_normalized
    H = source_normalized.T @ target_normalized
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Calculate translation
    t = target_centroid - scale * R @ source_centroid
    
    return scale, R, t


def add_glassesv2(img, img_lmk, glasses_mask, glasses_color, glasses_lmk, glasses_scale=1, frame_opacity=1):
    """
    [Joe Lee]
    * Add glasses (our custom dataset) given headpose of subject.
    * Augmentor changed from taking camera_id as input to headpose of subject.
    * Need to be consistent that glasses_lmk are inputted as *np.array of dim (4,2)

    Args:
        - img: input image (224 x 224) from dataset
        - img_lmk: mediapipe landmarks [443, 450, 223, 230] of input image
        - glasses_mask: rgb mask of glasses with frame(blue), left lens(green), right lens(red)
        - glasses_color: actual image of the glasses with 7 colors 
        - glasses_lmk: mediapipe landmarks [443, 450, 223, 230] of image used from GlassesGAN

    Returns:
        - result: (224 x 224) image of input image + glasses augmented
        - transformed_mask: transformed 3-color mask of the glasses for adding reflection
    """

    # Get mask of frame from channel 0 (Blue)
    frame_mask = glasses_mask[:, :, 0].astype(np.uint8)
    frame_mask = np.repeat(frame_mask[:, :, np.newaxis], 3, axis=2)

    s, R, t = find_similarity_transform(glasses_lmk[0], img_lmk, glasses_scale) # lmt_dst = lmk of faces
    A = s * R
    trans = np.hstack([A, t.reshape(2, 1)]) 
    height, width = img.shape[:2]

    # -------------------- cv2.resize() is not used here as it drops image quality ------------------
    # glasses_color = cv2.resize(glasses_color, (height, width), cv2.INTER_AREA)
    # frame_mask = cv2.resize(frame_mask, (height, width), cv2.INTER_AREA)
    
    # Resize glasses to input image using imutils
    glasses_color = imutils.resize(glasses_color, width=width)
    frame_mask = imutils.resize(frame_mask, width=width) 
    glasses_mask = imutils.resize(glasses_mask, width=width) # needed for reflection

    # just go through frame - same code as original add_glasses
    trans[0,2], trans[1,2] = trans[0,2] * height, trans[1,2] * width
    transformed_image = cv2.warpAffine(glasses_color, trans, (width, height)) 
    transformed_frame = cv2.warpAffine(frame_mask, trans, (width, height))  
    transformed_mask = cv2.warpAffine(glasses_mask, trans, (width, height))  

    result = img.astype(np.float32) * (1 - (transformed_frame*frame_opacity)/255) + transformed_image*frame_opacity

    return result.astype(np.uint8), transformed_mask

# def add_padding(img, scale=1.25): # Deprecated / only needed for MP-based headpose estimator
#     '''
#     Padding necessary for mediapipe to work on detecting landmarks where not enough background/face cropped in 
#     Tested on:
#         - Xgaze: scale=1.5
#     '''
#     H, W = img.shape[:2]

#     new_W = int(W * scale)
#     new_H = int(H * scale)

#     delta_W = new_W - W
#     delta_H = new_H - H

#     top = delta_H // 2
#     bottom = delta_H - top
#     left = delta_W // 2
#     right = delta_W - left

#     # Green background for xgaze 
#     border_color = [255, 255, 255]

#     padded_img = cv2.copyMakeBorder(
#         img, 
#         top, 
#         bottom, 
#         left, 
#         right, 
#         cv2.BORDER_CONSTANT, 
#         value=border_color
#     )

#     return padded_img

def get_glasses_pitch_yaw(pitch_lst, yaw_lst, subject_hp):
    '''
    [Joe Lee]
    Args:
        - subject_hp : headpose of the face; in gt file, key: 'tf_headpose'

    Returns:
        - is_valid (bool): if mediapipe doesn't detect landmarks -> return False, else returns True
        - pitch_bin_id (int)
        - yaw_bin_id (int)

        * pitch & yaw bin ids are used for glasses selection 
        * pitch bin ids: [0,1,2,3,4] -> (corresponds to) -> [-30, -15, 0, 15, 30]
        * yaw bin ids: [0,1,2,3,4,5] -> (corresponds to) -> [0, 9, 18, 27, 36, 45]
    '''
    subject_headpose = np.array([subject_hp[0], -1 * subject_hp[1], 0])

    rot_subject_quat = Rotation.from_euler('xyz', subject_headpose).as_quat()

    glasses_headposes = []
    glasses_headposes_quat = []
    for pitch in pitch_lst:
        for yaw in yaw_lst:
            glasses_headposes.append(np.array([pitch, yaw, 0]))
    
    glasses_headposes_quat = Rotation.from_euler('xyz', np.array(np.radians(glasses_headposes))).as_quat()

    # dot_prod is between -1.0 and 1.0 where closer to 0 means it is similar absolute values further away -> more different
    dot_prod = np.abs(np.dot(rot_subject_quat, glasses_headposes_quat.T))

    # Calculate angular distance to all points, and find the index
    angular_distances_rad = 2 * np.arccos(dot_prod)
    angular_distances_deg = np.degrees(angular_distances_rad)

    most_similar_id = np.argmin(angular_distances_deg)

    pitch_val, yaw_val = (glasses_headposes[most_similar_id])[0],  (glasses_headposes[most_similar_id])[1]

    # print(f'Assigned to {pitch_lst.index(pitch_val)} ({(pitch_val)} degrees) | {yaw_lst.index(yaw_val)} ({(yaw_val)} degrees)')

    return pitch_val, yaw_val