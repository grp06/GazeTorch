'''
Originally from GitHub
https://github.com/crisie/RecurrentGaze/blob/master/generate_training_CS.py
Revised by Mehdi.
Revised by Julia that generates the same number of CS images as RUDA (6487)
(removing eyeball rotation constraints and only select those annotated as 'OK')
Futher revised by Julia 2023/04/04 to match Beihang GazeHub's data normalization method

01/24/2026 - Joe: currently commented out to only return gaze/headpose. If needed, uncomment other parts as they have not changed from Julia's version.
'''

EYEDIAP_path = "/data/eyediap"
import os
import cv2
import sys
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.join(EYEDIAP_path, 'Scripts'))
from data_utils import *
from eyediap_utils import *

import matplotlib.pyplot as plt
plt.figure(figsize=(30, 30))
info_dict = {
    'fn': [],
    'gaze': [],
    'session': [],
    'fold_id': [],
    'headpose': [],
    'subject_cid': []
}
num_saved_data = 0 
save_frames_every = 15

def draw_gaze(image_in, g3d, thickness=2, color=(0, 255, 255)):
    """Draw 3D gaze angle starting from image center."""
    pitchyaw = g3d if len(g3d)==2 else xyz_to_pitchyaw(g3d)
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    h, w, _ = image_out.shape
    length = w / 2.0
    pos = (int(w / 2.0), int(h / 2.0))

    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    # practical length of the vector = sqrt(1-z^2)
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out


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



# Exp 5 refers to the Continuous Screen EYEDIAP subset
mod = 'CS'
type = ['S', 'M']  #  Change this accordingly

# Rotation matrix to convert CCS -> WCS
Rw = [[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
# Translation vector to convert CCS -> WCS
tw = [0., 0., 1.]
Rw = np.array(Rw)
tw = np.array(tw).reshape((3, 1))

for t in type:
    # Session selection (from EYEDIAP scripts)
    sessions = []
    for P in range(1,17):
        if P < 12 or P > 13:
            sessions.append(str(P) + "_A_CS_" + t)

    for session in sessions:
        print("Session: ", session)
        # NOTE: this may not work depending on session. Change accordingly.
        #session_str = get_session_string(session_num)
        session_str = session
        if ('7_A_CS' in session) or ('10_A_CS' in session) or ('15_A_CS' in session):
            fold_id = 0
        elif ('2_A_CS' in session) or ('4_A_CS' in session) or ('8_A_CS' in session):
            fold_id = 1
        elif ('3_A_CS' in session) or ('6_A_CS' in session) or ('16_A_CS' in session):
            fold_id = 2
        elif ('1_A_CS' in session) or ('5_A_CS' in session) or ('9_A_CS' in session):
            fold_id = 3
        elif ('14_A_CS' in session) or ('11_A_CS' in session):
            fold_id = 4
        else:
            raise ValueError('Cannot recognize the fold for session {}'.format(session))

        # EYEDIAP data path
        head_track_file = os.path.join(EYEDIAP_path, 'Data', session_str, 'head_pose.txt')
        screen_track_file = os.path.join(EYEDIAP_path, 'Data', session_str, 'screen_coordinates.txt')
        eye_track_file = os.path.join(EYEDIAP_path, 'Data', session_str, 'eye_tracking.txt')
        vga_calibration = os.path.join(EYEDIAP_path, 'Data', session_str, 'rgb_vga_calibration.txt')

        valid_file = os.path.join(EYEDIAP_path, 'Annotations', 'GazeState', 'GazeStateExport', 'Data',
                                  session_str,
                                  'gaze_state.txt')

        # Video frames data path
        video_path = os.path.join(EYEDIAP_path, 'Data', session_str, 'rgb_vga.mov')
        frames = get_frames(video_path)

        # If annotations on valid frames are available, then these are loaded and taken into account
        valid_frames = None
        if os.path.exists(valid_file):
            valid_frames = np.genfromtxt(valid_file, delimiter='\t', dtype=None)

        # Read the files with the frame-by-frame tracking parameters
        screen_track = read_screen_track_file(screen_track_file)
        head_track = read_head_track_file(head_track_file)

        landmarks = np.load(os.path.join(EYEDIAP_path, 'landmarks_3d', session_str, 'landmakrs_vga.npy'), allow_pickle=True)
        # Read calibration file
        calib = readCalibrationFile(vga_calibration)

        # Read eye track file
        with open(eye_track_file) as infile:
            eye_track = infile.readlines()

        # Containers for valid frames
        # -> Changed from np.bool to bool as np.bool is deprecated.
        valid_head = np.zeros(screen_track.shape[0], dtype=bool)
        valid_screen = np.zeros(screen_track.shape[0], dtype=bool)
        valid_gaze = np.zeros(screen_track.shape[0], dtype=bool)
        valid_face = np.zeros(screen_track.shape[0], dtype=bool)

        valid = False

        for frameIndex in range(len(head_track[0])):
            print("...................")
            print("Frame: ", frameIndex)
            if frameIndex % save_frames_every != 0:
                print("Skipping the frame ...")
                continue
            # Read the parameters for the current frame
            # Screen coordinates in CCS
            screen_pos_CCS = screen_track[frameIndex, :].reshape(3,)

            # Head Rotation Matrix to convert HCS -> WCS
            R = head_track[0][frameIndex, :, :]

            # Head Rotation Matrix to convert HCS -> CCS
            R_CCS = np.dot(Rw, R)

            # Head Translation to convert HCS -> WCS
            T = head_track[1][frameIndex, :].reshape(3, 1)

            # valid_head[frameIndex] = np.sum(T) != 0.0
            # valid_screen[frameIndex] = np.sum(screen_pos_CCS) != 0
            # Only use images annotated as 'OK', same as RUDA rotational consistency paper
            # valid_gaze[frameIndex] = valid_frames[frameIndex][1] == (b'OK')
            # valid_face[frameIndex] = np.sum(landmarks[frameIndex]) > 0.0

            # valid = valid_head[frameIndex] and valid_screen[frameIndex] and valid_face[frameIndex] and valid_gaze[frameIndex]
            # print("Valid: ", valid)

            # if not valid:
            #     continue
            
            # 3D eyeball centers in WCS
            # eyeball_info = eye_track[frameIndex + 1]
            # eyeball_info = list(map(eval, eyeball_info.strip().split(";")))
            # if len(eyeball_info) != 19:
            #     continue
            # eyeball_info = np.array(eyeball_info)
            # left3d = eyeball_info[13:16]
            # right3d = eyeball_info[16:19]
            # eyeball_centers_WCS = np.array([left3d, right3d]).reshape(2,3).transpose()

            # # 3D eyeball centers in CCS [left right]
            # eyeball_centers_CCS = np.dot(Rw.transpose(), eyeball_centers_WCS) - np.dot(Rw.transpose(), tw)
            # # Avg of 3D eyeball center vectors of left and right eyeballs in CCS
            # eyeball_centers_CCS_avg = np.mean(eyeball_centers_CCS.transpose(), axis=0)

            # Face normalization
            # Head vector in HCS
            head = np.array([[0.0], [0.0], [0.0]]).reshape(1, 3)

            head_center_WCS = np.dot(head, R.transpose()) + T.transpose()
            head_center_CCS = np.dot(Rw.transpose(), head_center_WCS.transpose()) - np.dot(Rw.transpose(), tw)
            head_center_CCS = head_center_CCS.transpose()

            # Face center in CCS, avg of head_center and avg position of two eyeballs
            # * 1000 for normalization function
            mean_face_CCS = (head_center_CCS[0]) * 1000

            face_roi_size = [224, 224]
            face_patch_conv, face_patch_warp, face_patch_gaze = get_normalized_data(mean_face_CCS, R_CCS, face_roi_size,
                                                                                    calib)
            normalized_face = warp_image(frames[frameIndex], face_patch_warp, (224, 224))
            # Normalize gaze vector using rotation matrix (R in normalization function)
            gaze_vector_CCS = np.dot(face_patch_gaze, (screen_pos_CCS * 1000 - mean_face_CCS))
            gaze_vector_CCS = gaze_vector_CCS / np.linalg.norm(gaze_vector_CCS)
            gaze_angles_CCS = vector2angles(gaze_vector_CCS)

            # Normalize head rotation vector using M in normalization function
            head_vector_CCS = cv2.Rodrigues(np.dot(face_patch_conv, R_CCS))[0].T[0]
            head_angle_CCS = vector2angles_head(head_vector_CCS)

            # both gaze angle and head angle are yaw, pitch so we reverse them to be as [pitch, yaw]
            gaze_angles_CCS = gaze_angles_CCS[:, ::-1]
            head_angle_CCS = head_angle_CCS[:, ::-1]
            
            cv2.imwrite(EYEDIAP_path + '/251222_face224_cs_julia/' + '{}_{}_{}.png'.format(session, fold_id, frameIndex), normalized_face[:, :, ::-1])
            info_dict['fn'].extend(['{}_{}_{}'.format(session, fold_id, frameIndex)])
            info_dict['session'].extend([session])
            info_dict['gaze'].extend(gaze_angles_CCS)
            info_dict['headpose'].extend(head_angle_CCS)
            info_dict['fold_id'].extend([fold_id])
            info_dict['subject_cid'].extend([int(session.split('_')[0])-1])
            num_saved_data += 1
            print('Num Data: {}'.format(num_saved_data))

            if num_saved_data <= 100:
                img = np.copy(normalized_face[:, :, ::-1])
                draw_out = draw_gaze(img, gaze_angles_CCS[0, :])
                plt.subplot(10, 10, num_saved_data)
                plt.imshow(draw_out[:, :, ::-1])
                plt.title('{}'.format(gaze_angles_CCS * 180/ np.pi))
                plt.axis('off')
            if num_saved_data == 100:
                plt.savefig(os.path.join(EYEDIAP_path, 'samples_cs_bh.png'))


np.save(EYEDIAP_path + "/251222_face224_gt_cs_julia.npy", info_dict)
