import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt
from tools.utils import draw_gaze
plt.figure(figsize=(30, 30))

GAZE360_DIR = '/data/gaze360'

info_dict = {
    'fn': [],
    'gaze': [],
    'gaze_ecs': [],
    'recording': [],
    'subject_cid': [],
    'frame': [],
    'split': []
}

def CropFaceImg(img, head_bbox, cropped_bbox):
    bbox =np.array([ (cropped_bbox[0] - head_bbox[0])/head_bbox[2],
              (cropped_bbox[1] - head_bbox[1])/head_bbox[3],
              cropped_bbox[2] / head_bbox[2],
              cropped_bbox[3] / head_bbox[3]])

    size = np.array([img.shape[1], img.shape[0]])

    bbox_pixel = np.concatenate([bbox[:2] * size, bbox[2:] * size]).astype("int")

    # Find the image center and crop head images with length = max(weight, height)
    center = np.array([bbox_pixel[0]+bbox_pixel[2]//2, bbox_pixel[1]+bbox_pixel[3]//2])

    length = int(max(bbox_pixel[2], bbox_pixel[3])/2) 

    center[0] = max(center[0], length)
    center[1] = max(center[1], length)

    result = img[(center[1] - length) : (center[1] + length),
                (center[0] - length) : (center[0] + length)] 

    result = cv2.resize(result, (224, 224))
    return result

def load_gaze_dict():
    gaze_dict = dict()
    for text_files in ['train.txt', 'validation.txt', 'test.txt']:
        lines = open(os.path.join(GAZE360_DIR, text_files), 'r').readlines()
        for line in tqdm(lines):
            address = line.split(' ')[0]
            gaze_x = float(line.split(' ')[1])
            gaze_y = float(line.split(' ')[2])
            gaze_z = float(line.split(' ')[3])    
            yaw = np.arctan2(gaze_x, -gaze_z)
            pitch = np.arcsin(gaze_y)
            gaze_dict[address] = (text_files.split('.')[0], pitch, yaw)
    
    return gaze_dict
    

if __name__ == "__main__":

    normalized_gaze_dict = load_gaze_dict()
    msg = sio.loadmat(os.path.join(GAZE360_DIR, "metadata.mat"))
    recordings = msg["recordings"]
    gazes = msg["gaze_dir"]
    head_bbox = msg["person_head_bbox"]
    face_bbox = msg["person_face_bbox"]
    lefteye_bbox = msg["person_eye_left_bbox"]
    righteye_bbox = msg["person_eye_right_bbox"]
    splits = msg["splits"]
    split_index = msg["split"]
    recording_index = msg["recording"]
    person_index = msg["person_identity"]
    frame_index = msg["frame"]
    total_num = recording_index.shape[1]
    headposes = msg['person_eyes2d']

    n_saved = 0
    n_skipped = 0
    # process each image
    for i in tqdm(range(total_num)):
        im_path = os.path.join(GAZE360_DIR, "imgs",
            recordings[0, recording_index[0, i]][0],
            "head", '%06d' % person_index[0, i],
            '%06d.jpg' % frame_index[0, i])
        im_path_local = os.path.join(recordings[0, recording_index[0, i]][0],
            "head", '%06d' % person_index[0, i], '%06d.jpg' % frame_index[0, i])

        category = splits[0, split_index[0, i]][0]
        #ECS: Eye coordinate system
        gaze_in_ECS = gazes[i]
        yaw_ecs = np.arctan2(gaze_in_ECS[0], -gaze_in_ECS[2])
        pitch_ecs = np.arcsin(gaze_in_ECS[1])

        if im_path_local not in normalized_gaze_dict.keys():
            continue

        if (face_bbox[i] == np.array([-1, -1, -1, -1])).all():
            n_skipped += 1
            continue


        img = cv2.imread(im_path)
        face = CropFaceImg(img, head_bbox[i], face_bbox[i])
        pitch = normalized_gaze_dict[im_path_local][1]
        yaw = normalized_gaze_dict[im_path_local][2]

        recording = recordings[0, recording_index[0, i]][0]
        subject_cid = person_index[0, i]
        fn = '{}_{}_{}'.format(recording, subject_cid, i)
 
        cv2.imwrite(os.path.join(GAZE360_DIR, 'face_224', fn + '.png'), face)
        info_dict['fn'].extend([fn])
        info_dict['gaze'].append([pitch, yaw])
        info_dict['gaze_ecs'].append([pitch_ecs, yaw_ecs])
        info_dict['recording'].extend([recording])
        info_dict['frame'].extend([frame_index[0, i]])
        info_dict['subject_cid'].extend([subject_cid])
        info_dict['split'].extend([normalized_gaze_dict[im_path_local][0]])
        n_saved += 1

        if n_saved <= 100:
            draw_out = draw_gaze(face, [pitch, yaw])
            plt.subplot(10, 10, n_saved)
            plt.imshow(draw_out[:, :, ::-1])
            plt.title('[{}, {}]'.format(round(pitch * 180/np.pi, 2), round(yaw * 180/np.pi, 2)))
            plt.axis('off')
        if n_saved == 100:
            plt.savefig(os.path.join(GAZE360_DIR, 'samples.png'))

info_dict['gaze'] = np.array(info_dict['gaze'])
info_dict['gaze_ecs'] = np.array(info_dict['gaze_ecs'])

print('num saved data: ', n_saved)
print('num skipped: ', n_skipped)
np.save(os.path.join(GAZE360_DIR, 'gaze360'), info_dict)
