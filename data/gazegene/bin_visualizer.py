import os
import cv2
import torch
import random
import numpy as np
import tqdm

'''
Author: Joe
Visualization of P/Y for each bin in dataset
'''

PATH_ID = "/store/GazeGene/valid_ids.npy"
PATH_GAZE = '/store/GazeGene/gazegene_gt_conf.npy'
FACE_ID = "/store/GazeGene/face_224"

def getPitchYawCid(gaze, nbins_pitch=9, nbins_yaw=13, bin_size=4, pitch_bias=-12):
    # Copied directly from gazegene.py
    pitch_min = pitch_bias - nbins_pitch *    bin_size / 2
    pitch_cid = np.clip((gaze[0] * 180/np.pi - pitch_min) //  bin_size, 0,  nbins_pitch - 1).astype(int)
    yaw_min = 0 -   nbins_yaw *    bin_size / 2
    yaw_cid = np.clip((gaze[1] * 180/np.pi - yaw_min) //  bin_size, 0,  nbins_yaw - 1).astype(int)

    return pitch_cid, yaw_cid

def loadSubjectBins(id_path, gaze_path):
    id_data = np.load(id_path)
    gaze_data = np.load(gaze_path, allow_pickle=True).item()
    # dict_keys(['fn', 'subject_cid', 'cam_id', 'gaze', 'headpose', 'wE_L', 'wI_L', 'wE_R', 'wI_R'])

    subject_range = range(1, 57) # 58 subjects in gazegene
    bin_range = [(x, y) for x in range(10) for y in range(14)]
    subject_bins = {subject: {bin: [] for bin in bin_range} for subject in subject_range}
    # e.g) subject_bins(nested dictionary)-> {subject_id : {(x,y): [list of img_ids]}} 
    
    for id in id_data:
        if np.abs(gaze_data['headpose'][id][2]) > np.pi/18 or np.abs(gaze_data['headpose'][id][2]) < np.pi/36: # if headpose* roll is greater than pi/18, ignore
            continue
        pitch_yaw = getPitchYawCid(gaze_data['gaze'][id])
        subject_bins[int(gaze_data['subject_cid'][id])][pitch_yaw].append(id)
    
    return subject_bins

def emptyGrid(nbins_pitch=9, nbins_yaw=13):
    # Initialize empty background 
    bg_w = nbins_pitch * 224 # original images are 224*224
    bg_h = nbins_yaw * 224
    bg = np.zeros((bg_w, bg_h, 3), dtype=np.uint8)
    return bg

def addImagetoBin(image_id, bin_row, bin_col, background):
    # place image on the background depening on bin row/col
    image = cv2.imread(f'{FACE_ID}/{image_id[0]}.png')
    x = bin_row * 224
    y = bin_col * 224
    
    background[x: x + 224, y: y + 224, :] = image

def sampleImages(subject_id, subject_bins, background):
    # randomly sample from each subject
    subject_bin = subject_bins[subject_id]
    for coords, id_list in subject_bin.items():
        if len(id_list) >= 1:
            addImagetoBin(random.sample(id_list, 1), coords[0], coords[1], background) # -> id, gaze_pitch, gaze_yaw, background
    

subject_bins = loadSubjectBins(PATH_ID, PATH_GAZE)

for subject_id in range(1,57):
    background = emptyGrid() # init empty background
    sampleImages(subject_id, subject_bins, background)
    cv2.imwrite(f'/store/joelee217/bins/{subject_id}_bin.png', background)
    print(f'subject {subject_id} completed!')