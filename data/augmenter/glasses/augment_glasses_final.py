'''
Kept for reference. Currently, glasses_augmenter.py is used.
'''

import numpy as np
import cv2
import json
import random
import imutils

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


class Aug_Glasses:
    def __init__(self, img, mask, lmk):
        with open(lmk, 'r') as file:
            self.glass_lmk = json.load(file)
        self.glass_lmk = list(self.glass_lmk.values())
        # self.glass_img = np.load(None, allow_pickle=True)
        self.glass_img = cv2.imread(img)
        self.glass_mask = cv2.imread(mask)

    def add_glasses(self, lmk_dst, img_dst):
        s, R, t = find_similarity_transform(self.glass_lmk, lmk_dst)
        A = s * R
        trans = np.hstack([A, t.reshape(2, 1)]) 

        height, width = img_dst.shape[:2]
        self.glass_img = cv2.resize(self.glass_img, (height, width)) # You can move this to init and use config parameters for the size
        self.glass_mask = cv2.resize(self.glass_mask, (height, width)) # You can move this to init and use config parameters for the size

        trans[0,2], trans[1,2] = trans[0,2] * height, trans[1,2] * width
        transformed_image = cv2.warpAffine(self.glass_img, trans, (width, height)) 
        transformed_mask = cv2.warpAffine(self.glass_mask, trans, (width, height))       
       
        # img_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2RGB)
        result = img_dst * (1 - transformed_mask/255) + transformed_image 
        return result


def add_glasses_amin(img, mask, lmk, lmk_dst, img_dst, glasses_scale=1.0, frame_opacity=1):
    
    glass_lmk = lmk
    glass_img = img

    frame_mask = np.stack((mask[:,:, 0],) * 3, axis=-1) # Only the blue channel

    pure_mask = mask#[:,:, 1][:,:,None] + mask[:,:, 2][:,:,None]

    s, R, t = find_similarity_transform(glass_lmk, lmk_dst, glasses_scale) # lmt_dst = lmk of faces
    A = s * R
    trans = np.hstack([A, t.reshape(2, 1)]) 

    height, width = img_dst.shape[:2]
    glass_img = cv2.resize(glass_img, (height, width)) # You can move this to init and use config parameters for the size

    pure_mask = cv2.resize(pure_mask, (height, width))

    frame_mask = cv2.resize(frame_mask, (height, width)) # You can move this to init and use config parameters for the size
    # just go through frame
    trans[0,2], trans[1,2] = trans[0,2] * height, trans[1,2] * width
    transformed_image = cv2.warpAffine(glass_img, trans, (width, height)) 
    transformed_frame = cv2.warpAffine(frame_mask, trans, (width, height))  

    pure_mask = cv2.warpAffine(pure_mask, trans, (width, height)) 
    result = img_dst * (1 - (transformed_frame*frame_opacity)/255) + transformed_image*frame_opacity
    # result = img_dst * (1 - (transformed_glass_right*glasses_opacity_right)/255 - (transformed_glass_left*glasses_opacity_left)/255 - (transformed_frame*frame_opacity)/255) + transformed_glass_right*glasses_opacity_right + transformed_glass_left*glasses_opacity_left + transformed_image*frame_opacity
    result = result.astype(np.uint8)
    return result, pure_mask


def add_glassesv2(img, img_lmk, glasses_mask, glasses_color, glasses_lmk, glasses_scale=1, frame_opacity=1):
    """
    [Joe Lee]
    * Add glasses (our custom dataset) given headpose of subject.
    * Augmentor changed from taking camera_id as input to headpose of subject.

    Args:
        - img: input image (224 x 224) from dataset
        - img_lmk: mediapipe landmarks [223, 230, 443, 450] of input image
        - glasses_mask: rgb mask of glasses with frame(blue), left lens(green), right lens(red)
        - glasses_color: actual image of the glasses with 7 colors 
        - glasses_lmk: mediapipe landmarks [223, 230, 443, 450] of image used from GlassesGAN

    Returns:
        - result: (224 x 224) image of input image + glasses augmented
    """

    # Get mask of frame from channel 0 (Blue)
    frame_mask = glasses_mask[:, :, 0].astype(np.uint8)
    frame_mask = np.repeat(frame_mask[:, :, np.newaxis], 3, axis=2)

    # Read in mediapipe landmarks for alignment
    img_lmk = np.array(list(img_lmk.item().values()))
    glasses_lmk = np.array(list(glasses_lmk.item().values()))

    s, R, t = find_similarity_transform(glasses_lmk, img_lmk, glasses_scale) # lmt_dst = lmk of faces
    A = s * R
    trans = np.hstack([A, t.reshape(2, 1)]) 
    height, width = img.shape[:2]

    # -------------------- cv2.resize() is deprecated as it drops image quality ------------------
    # glasses_color = cv2.resize(glasses_color, (height, width), cv2.INTER_AREA)
    # frame_mask = cv2.resize(frame_mask, (height, width), cv2.INTER_AREA)
    
    # Resize glasses to input image using imutils
    glasses_color = imutils.resize(glasses_color, width=width)
    frame_mask = imutils.resize(frame_mask, width=width) 

    # just go through frame
    trans[0,2], trans[1,2] = trans[0,2] * height, trans[1,2] * width
    transformed_image = cv2.warpAffine(glasses_color, trans, (width, height)) 
    transformed_frame = cv2.warpAffine(frame_mask, trans, (width, height))  

    result = img.astype(np.float32) * (1 - (transformed_frame*frame_opacity)/255) + transformed_image*frame_opacity

    return result.astype(np.uint8)

def main():
    img = cv2.imread('/store/joelee217/GlassesGAN_release/img_in/xgaze/229320-cam0.png')
    img_lmk = np.load('/store/joelee217/GlassesGAN_release/img_in/lmks/xgaze_test.npy', allow_pickle=True)

    glasses_rgb = cv2.imread('/store/joelee217/GlassesGAN_release/img_out/zhenhao4/zhenhao4_rgb.png')
    glasses_color = cv2.imread('/store/joelee217/GlassesGAN_release/img_out/zhenhao4/zhenhao4_red.png')
    glasses_lmk = np.load('/store/joelee217/GlassesGAN_release/img_in/lmks/zhenhao0.npy', allow_pickle=True)

    res = add_glassesv2(img, img_lmk, glasses_rgb, glasses_color, glasses_lmk)

    cv2.imwrite('/store/joelee217/GlassesGAN_release/img_out/xgaze_aug/test2.png', res)

# def main():
#     source = '/store/afadaeinejad/projects/GlassesGAN_release/samples'
#     glasses_id = 92125
#     glass_lmk_path = f"{source}/landmarks4/{glasses_id}.json"
#     glass_img_path = f"{source}/glasses3/{glasses_id}.jpg"
#     glass_mask_path = f"{source}/mask4/{glasses_id}.jpg"
    
#     label = np.load('/data/xgaze/448/lmk_224.npy', allow_pickle=True)
#     aug = Aug_Glasses(glass_img_path, glass_mask_path, glass_lmk_path)
#     for i in range(25):
#         if not label[i]:
#             continue
#         image = cv2.imread(f"/data/xgaze/448/face_224/{i}.jpg")
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         res = aug.add_glasses(label[i], image_rgb)
#         cv2.imwrite('test.png', res)



if __name__ == '__main__':
    main()