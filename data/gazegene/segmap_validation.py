import os
import cv2
import random
import numpy as np
from tqdm import tqdm 
from pathlib import Path
import concurrent.futures
import matplotlib.pyplot as plt
import logging

'''
Author: Joe
Validate Seg maps using IOU
'''

PATH_GAZE = '/store/GazeGene/gazegene_gt_conf_v2.npy'
PATH_GAZE_PLUS = '/store/GazeGene/gazegene_gt_conf_v2+.npy'
PATH_GAZE_PLUS_IOU = '/store/GazeGene/gazegene_gt_conf_v2+_iouv2.npy'
PATH_GAZE_PLUS_IOUV2 = "/store/joelee217/gazegene_gt_conf_v2+_iouv2_eye_iris_check.npy"

PATH_SEG = '/store/GazeGene/seg'
PATH_FACE224 = '/store/GazeGene/face_224'
PATH_IDS = '/store/GazeGene/id_list_gazegene.npy'
PATH_VALID_IDS = '/store/GazeGene/valid_ids.npy'
PATH_ROLL_ADJ_IDS = '/store/GazeGene/valid_ids_roll_pi36.npy'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_gt_conf(gaze_dataset, switch):
    if switch: 
        np.save('/store/joelee217/gazegene_gt_conf_v2+_iouv2_eye_iris_check.npy', gaze_dataset)

# Calculate IoU based on (eye ∧ iris) / (eye V iris)
def calculateIOU(mask_eye, mask_iris):
    intersection = np.logical_and(mask_eye, mask_iris).sum()
    union = np.logical_or(mask_eye, mask_iris).sum()
    
    if union == 0:
        return 0.0  
    
    iou = intersection / union

    return iou

# Calculate IoU based on (iris) / (eye V iris)
def calculateIOUV2(mask_eye, mask_iris):
    union = np.logical_or(mask_eye, mask_iris).sum()
    iris = np.logical_and(union, mask_iris).sum()

    if union == 0:
        return 0.0  
    iou = iris / union

    return iou

def getIOU(sample_id):
    sample_str = f'{sample_id}.png'
    # if path for segmap exists, get segmap
    if Path(os.path.join(PATH_SEG, sample_str)).exists(): 
        sample_segmap = cv2.imread(os.path.join(PATH_SEG, sample_str), cv2.IMREAD_UNCHANGED) # <-- needed to read 4D .png file
    '''
    - segmap <- (224,224,4) ; 4D trinary mask
    - (background: 0), (eye: 1), (eyelid : 2) 
    - sample_segmap[:, :, 0] <- left eye
    - sample_segmap[:, :, 1] <- left iris
    - sample_segmap[:, :, 2] <- right eye
    - sample_segmap[:, :, 3] <- right iris
    '''
    left_iou, right_iou = 0, 0

    for i in range(0, 4, 2):
        mask_eye = sample_segmap[:, :, i]
        mask_eye[mask_eye==2] = 0 # ignore the eyelids
        mask_iris = sample_segmap[:, :, i + 1]

        if i == 0: # if Left
            left_iou = calculateIOUV2(mask_eye, mask_iris)
        else: # if right
            right_iou = calculateIOUV2(mask_eye, mask_iris)

    return (left_iou, right_iou)

def processIOU(sample):
    # track logging as tqdm doesn't work with concurrent.futures
    if sample % 10000 == 0:
        logging.log(f'{sample}/1008000')
    
    (left_iou, right_iou) = getIOU(sample)
    
    return (sample, left_iou, right_iou)

def filterInvalidIOU(sample, threshold=0.2):
    '''
    filter if iris area is larger than eye area -> invalid sample id
    Takes in sample, return sample id if it is valid
    Note: this goes through PATH_GAZE_PLUS_IOU so the IOU_L and IOU_R already exists for each sample
    '''
    # read paths for segmaps
    sample_str = f'{sample}.png'
    seg_path = Path(PATH_SEG) / sample_str
    
    # Check valid paths; 
    if not seg_path.exists():
        logging.info(f'Failed to find {seg_path}')
        return None

    # Read segmap
    seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)

    # --------- logic for filtering invalid segmaps ------------

    # if IoU is less than threshold for either L or R
    left_iou, right_iou = getIOU(sample)

    if left_iou < threshold or right_iou < threshold:
        return None

    # Check left and right iou
    mask_eye_left = seg[:, :, 0].copy()
    mask_eye_left[mask_eye_left == 2] = 0
    mask_iris_left = seg[:, :, 1]
    
    mask_eye_right = seg[:, :, 2].copy()
    mask_eye_right[mask_eye_right == 2] = 0
    mask_iris_right = seg[:, :, 3]

    # if iris is larger than eye or less than threshold
    if np.sum(mask_eye_left) < np.sum(mask_iris_left) or np.sum(mask_eye_right) < np.sum(mask_iris_right):
        return None

    # if passes all tests, 
    return sample

# Read in data
gaze_data = np.load(PATH_GAZE_PLUS_IOUV2, allow_pickle=True).item()
id_data = list(np.load(PATH_ROLL_ADJ_IDS, allow_pickle=True))

max_workers = os.cpu_count() * 4  
filtered_ids = []

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for sample in executor.map(filterInvalidIOU, tqdm(id_data)):
        try:
            if sample is not None:
                filtered_ids.append(sample)
        except Exception as e:
            logging.error(f'Error processing sample: {e}')

print(len(filtered_ids))
np.save('/store/joelee217/valid_ids_roll_adj_v2+.npy', filtered_ids)
# print(len(list(np.load('/store/GazeGene/valid_ids_v2.npy', allow_pickle=True))))
# print(len(list(np.load('/store/joelee217/valid_ids_v2+.npy', allow_pickle=True))))

'''  --------------------------- Visualizations ----------------------------------

def loadSegmaps(gaze_path, seg_path, img_path, id_path):
    gaze_data = np.load(gaze_path, allow_pickle=True).item()

    gaze_data['IOU_L'] = [None] * len(gaze_data['gaze'])
    gaze_data['IOU_R'] = [None] * len(gaze_data['gaze'])

    with concurrent.futures.ThreadPoolExecutor(max_workers=72*4) as executor:
        for sample in tqdm(range(250000, 250010)):
            sample_str = f'{sample}.png'
            confidence = (gaze_data['wE_L'][sample], gaze_data['wI_L'][sample], gaze_data['wE_R'][sample], gaze_data['wI_R'][sample])
            if Path(os.path.join(img_path, sample_str)).exists(): 
                sample_seg = cv2.imread(os.path.join(seg_path, sample_str), cv2.IMREAD_UNCHANGED)
                original_img = cv2.imread(os.path.join(img_path, sample_str))
                (left_iou, right_iou) = processIOU(sample_seg)
            else:
                (left_iou, right_iou) = (0.0, 0.0)
            # print(f'{sample} -> {confidence}')
            print(f'{sample} -> {left_iou, right_iou}')
            gaze_data['IOU_L'][sample] = left_iou
            gaze_data['IOU_R'][sample] = right_iou
        
    save_gt_conf(gaze_data, True)


def saveMask(sample_seg, original_img, fn, confidence):
    for i in range(4):
        mask = sample_seg[:, :, i]
        red_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        red_mask[:, :, 2] = mask
        red_mask_n = (red_mask / 2 * 255).astype(np.uint8)

        return_img = cv2.add(original_img, red_mask_n)
        # return_img = original_img + red_mask_n
        cv2.imwrite(f'/store/joelee217/seg/{fn.split(".png")[0]}_mask{i}_conf{confidence[i]}.png', return_img) 
    
def plotHist(dataset, name):
plt.hist(dataset, bins=50)  # bins=10 can be adjusted
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title(f'Histogram for {name}')
plt.ylim(0, 50000)
plt.savefig(f'/store/joelee217/{name}.png')
plt.close()

def visualizeIOU(sample):
    sample_str = f'{sample}.png'
    print(os.path.join(PATH_FACE224, sample_str))
    if Path(os.path.join(PATH_FACE224, sample_str)).exists(): 
        seg = cv2.imread(os.path.join(PATH_SEG, sample_str), cv2.IMREAD_UNCHANGED)
        original_img = cv2.imread(os.path.join(PATH_FACE224, sample_str))

        mask_eye = seg[:, :, 0]
        mask_eye[mask_eye==2] = 0 # ignore the eyelids
        
        mask_iris = seg[:, :, 1]

        mask_eye[mask_iris > 0] = 0

        iris_red = np.zeros((224,224,3))
        iris_red[:,:,2] = mask_iris
        
        eye_green = np.zeros((224,224,3))
        eye_green[:,:,1] = mask_eye

        mask_eye = np.repeat(mask_eye[:, :, np.newaxis], 3, axis=2)
        # mask_iris = np.repeat(mask_iris[:, :, np.newaxis], 3, axis=2)
        iou_mask = iris_red + eye_green
        # cv2.imwrite(f'/store/joelee217/iou_v2/{sample}.png', )
        cv2.imwrite(f'/store/joelee217/iou_v2/{sample_str}', original_img + iou_mask*255)

'''