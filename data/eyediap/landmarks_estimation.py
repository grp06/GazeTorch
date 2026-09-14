'''
LandmarksEstimation class from
https://github.com/crisie/RecurrentGaze/blob/master/landmarks_estimation.py
* relies on face_alignment library
'''

import face_alignment
from glob import glob
import os
import cv2
from tqdm import tqdm
import torch
import numpy as np

class LandmarksEstimation(object):
    """
    LandmarksEstimation class. Wrapper of Bulat et al 2017 method
    https://github.com/1adrianb/face-alignment
    """
    def __init__(self, model_type=face_alignment.LandmarksType._3D, device='gpu'):
        if device == 'gpu':
            self.landmarks_model = face_alignment.FaceAlignment(model_type, flip_input=False, device='cuda')
        elif device == 'cpu':
            self.landmarks_model = face_alignment.FaceAlignment(model_type, flip_input=False, device='cpu')
        else:
            raise RuntimeError("Device type not found")

    # TODO: check if gpu/cpu selector works properly
    @classmethod
    def get_model(cls):
        return LandmarksEstimation.landmarks_model


def get_frames(video_path):
    print('extracting frames ...')
    vidcap = cv2.VideoCapture(video_path)
    video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in tqdm(range(video_length)):
        _, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    return frames


def get_landmarks(list_frames, landmark_obj):
    frames_torch = torch.from_numpy(np.array(frames))
    frames_torch = frames_torch.permute(0, 3, 1, 2)
    print('extracting landmarks ...')
    batches = torch.split(frames_torch, 8)
    landmarks = []
    for i, batch in tqdm(enumerate(batches)):
        l = landmark_obj.landmarks_model.get_landmarks_from_batch(batch)
        landmarks.extend(l)

    return landmarks

if __name__ == "__main__":
    session_dirs = glob('/data/eyediap/Data/*')
    landmark_obj = LandmarksEstimation(device='gpu')
    type_video = 'vga'
    for session_dir in session_dirs:
        if os.path.exists('/data/eyediap/landmarks_3d/' + session_dir.split('/')[-1] + '/landmakrs_' + type_video + '.npy'):
            continue
        print('Session name: ', session_dir.split('/')[-1])
        os.makedirs('/data/eyediap/landmarks_3d/' + session_dir.split('/')[-1], exist_ok=True)        
        video_path = session_dir + '/rgb_' + type_video + '.mov'
        frames = get_frames(video_path)
        landmarks = get_landmarks(frames, landmark_obj)
        landmarks = np.array(landmarks)
        np.save('/data/eyediap/landmarks_3d/' + session_dir.split('/')[-1] + '/landmakrs_' + type_video + '.npy', landmarks)
