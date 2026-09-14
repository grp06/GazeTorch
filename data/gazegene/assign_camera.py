import numpy as np
from tqdm import tqdm

PATH_ID = '/store/GazeGene/id_list_gazegene.npy'
PATH_HEADPOSE = '/store/GazeGene/gazegene_gt_conf.npy'

def getCameraIdx(center):
    '''
    Get Idx of Camera (0 - 5) for GazeGene; corresponding to same format
    implemented for gc3d. Uses approximation to center coordinates and 
    returns closest index (position).
    '''

    headpose_data = np.load(PATH_HEADPOSE, allow_pickle=True).item()
    # dict_keys(['fn', 'subject_cid', 'cam_id', 'gaze', 'headpose', 'wE_L', 'wI_L', 'wE_R', 'wI_R'])

    headposes = np.array(headpose_data['headpose'])  # shape (N, M) - M dims (pitch, yaw, roll, etc.)

    # ['headpose'] in (P,Y,R) format; use pitch and yaw (indices 0 and 1)
    pitch_yaw = headposes[:, :2]  # shape (N, 2)

    # Initialize empty list to store cam data for each idx
    cam_data = [None] * len(pitch_yaw)

    # Vectorize loop to find closest center for each headpose vector
    for idx in tqdm(range(len(pitch_yaw))):
        # Compute distances from current headpose to all centers
        distances = np.linalg.norm(center - pitch_yaw[idx], axis=1)
        cam_data[idx] = np.argmin(distances)

    np.save('/store/GazeGene/gazegene_camera.npy', cam_data, allow_pickle=True)

if __name__ == '__main__':
    # Load center coords. predefined from XGaze
    center = np.load('center.npy', allow_pickle=True)
    getCameraIdx(center)
