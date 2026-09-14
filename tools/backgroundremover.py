import os
import cv2
import numpy as np
from tqdm import tqdm
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class BackgroundRemover():
    '''
    Author: Joe Lee 
    Removes background for GazeGene dataset using CIELAB space.
    Currently, the parameters are heuristic(experimented based on 32 samples):
    * blue_level: the level in B for CIELAB space that we consider as blue; the more negative the value, the more blue it is.
    * perc_blue: % of the edges that are considered blue based on *blue_level. If perc_blue is higher, we assume the background is largly blue.
    * threshold: The distance between the brightest background and all pixels in img.
    '''
    # Optional: Define kernel for opening/closing transforms
    def circular_kernel(radius=2, size=5):
        center = size // 2
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        return mask.astype(np.uint8)

    def find_darkest_point(img_lab):
        L_channel = img_lab[:, :, 0] # Extract L channel
        min_idx = np.unravel_index(np.argmin(L_channel), L_channel.shape)  # Find index of min L(Light)
        y, x = min_idx     
        return (y, x)
    
    def find_bluepixel(blue_list):
        L_channel = blue_list[:, 0] # Extract L channel
        max_idx = np.unravel_index(np.argmax(L_channel), L_channel.shape)  # Find index of max L(Light)
        return blue_list[max_idx]

    def findBackground(img):
        # Extract edge pixels
        top_edge = img[0:2, :]        # first two rows (top edge)
        bottom_edge = img[-2:, :]     # last two rows (bottom edge)
        left_edge = img[:, 0:2]       # first two columns (left edge)
        right_edge = img[:, -2:]      # last two columns (right edge)

        # (N, 3) array of all edge pixels concatenated
        edge_pixels = np.concatenate([top_edge.reshape(-1, top_edge.shape[2]), bottom_edge.reshape(-1, bottom_edge.shape[2]), 
                                      left_edge.reshape(-1, left_edge.shape[2]), right_edge.reshape(-1, right_edge.shape[2])], axis=0)

        blue_pixels = edge_pixels[edge_pixels[:, 2] < -5] # if B is less than -5 (blue part)
        
        if (len(blue_pixels)/len(edge_pixels)) > 0.2: # if 20% or more is blue in edge
            return BackgroundRemover.find_bluepixel(blue_pixels)
        else:
            # Darkest(lowest L) in image
            tgt = img[BackgroundRemover.find_darkest_point(img)]
            
            # Compute distances from tgt to each edge pixel
            dists = np.linalg.norm(edge_pixels - tgt, axis=1)

            # Find index of max distance
            max_idx = np.argmax(dists)
            return edge_pixels[max_idx]

    def findMask(img):
        # Define Empty Mask
        mask = np.zeros(img.shape[:2])
        # background: approximate background color assuming background is generally brighter / on edge
        background = BackgroundRemover.findBackground(img) # <- this is the value (3,1) of pixel not coords.

        # Compute the Euclidean distance from each pixel to the background
        distance = np.linalg.norm(img - background, axis=-1)

        # return mask
        # 1 where distance < (threshold; 15 is considered wide; use 4 for narrow; 8-10 for general) else 0
        mask = (distance < 15).astype(np.uint8) 

        return (mask[:,:, np.newaxis]).astype(np.float32) * 255
    
    def background_mediapipe(img):
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        BaseOptions = mp.tasks.BaseOptions
        ImageSegmenter = mp.tasks.vision.ImageSegmenter
        ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        model_path = '/store/joelee217/GazeTorch/data/gazegene/selfie_segmenter.tflite'

        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True)
        
        with ImageSegmenter.create_from_options(options) as segmenter:
            segmented_masks = segmenter.segment(mp_image)

        return segmented_masks.category_mask.numpy_view()


    def process_dir(source_dir, target_dir):
        # Parameters for the Guided Filter
        radius = 8 # Radius of the filter
        eps = 0.02 * 0.02 # Regularization parameter

        for fn in tqdm(os.listdir(source_dir)):

            start_time = time.time()

            # Process images from path
            path = os.path.join(source_dir, fn)
            rgb = cv2.imread(path)[:,:,::-1]
            rgb = rgb.astype(np.float32)
            rgb /= np.amax(rgb)

            # mask = BackgroundRemover.background_mediapipe(rgb)

            # Convert to LAB color space + get mask
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
            mask = BackgroundRemover.findMask(lab)

            totalLabels, label_ids, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8, cv2.CV_32S)
            max_label = np.argmax(stats[1:totalLabels, cv2.CC_STAT_AREA]) + 1

            # If background is greater than 40% of image,
            if (stats[max_label, cv2.CC_STAT_AREA]/(224*224)) > 0.4:
                # return empty mask
                mask = np.zeros(lab.shape[:2])
            else: 
                mask = (label_ids == max_label).astype('uint8') * 255

            # Optional: close/open morph. transforms
            # kernel = BackgroundRemover.circular_kernel()
            # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # opening = cv2.erode(mask, kernel)

            guided_filter = cv2.ximgproc.createGuidedFilter(lab, radius, eps)
            filtered = guided_filter.filter(mask.astype(np.uint8), dDepth=-1)
            
            path_target = os.path.join(target_dir, fn)
            cv2.imwrite(path_target, filtered)

            elapsed = time.time() - start_time
            print(f"Processed {fn} in {elapsed:.3f} seconds")

    def process_dir_concurrent(fn, source_dir, target_dir):
        # Parameters for the Guided Filter
        radius = 8 # Radius of the filter
        eps = 0.02 * 0.02 # Regularization parameter

        # Process images from path
        path = os.path.join(source_dir, fn)
        rgb = cv2.imread(path)[:,:,::-1]
        rgb = rgb.astype(np.float32)
        rgb /= np.amax(rgb)

        # Convert to LAB color space + get mask
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
        mask = BackgroundRemover.findMask(lab)

        totalLabels, label_ids, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8, cv2.CV_32S)
        max_label = np.argmax(stats[1:totalLabels, cv2.CC_STAT_AREA]) + 1

        # If background is greater than 40% of image,
        if (stats[max_label, cv2.CC_STAT_AREA]/(224*224)) > 0.4:
            return False
        else: 
            mask = (label_ids == max_label).astype('uint8') * 255

        guided_filter = cv2.ximgproc.createGuidedFilter(lab, radius, eps)
        filtered = guided_filter.filter(mask.astype(np.uint8), dDepth=-1)
        
        path_target = os.path.join(target_dir, fn)
        cv2.imwrite(path_target, filtered)
        return True

    def process_dir_concurrent_mp(fn):
        # Process images from path
        source_dir = "/data/gazecapture_3d/face"
        target_dir = "/data/gazecapture_3d/mp_backgrounds"
        fn = str(fn) + '.jpg'
        path = os.path.join(source_dir, fn)
        rgb = cv2.imread(path)[:,:,::-1]
        rgb = rgb.astype(np.float32)
        rgb /= np.amax(rgb)

        mask = BackgroundRemover.background_mediapipe(rgb)

        totalLabels, label_ids, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8, cv2.CV_32S)
        if totalLabels > 1:
            max_label = np.argmax(stats[1:totalLabels, cv2.CC_STAT_AREA]) + 1
        else:
            max_label = 0
        mask = (label_ids == max_label).astype('uint8') * 255

        path_target = os.path.join(target_dir, fn)
        cv2.imwrite(path_target, mask)
        return True

# source_dir = "/store/joelee217/GazeTorch/glasses_test_result/gazegene_test_joe"
# target_dir = "/store/joelee217/background_removal/photoshop"
# BackgroundRemover.process_dir_concurrent_mp('', source_dir, target_dir)