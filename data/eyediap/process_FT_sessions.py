'''
Originally from GitHub
https://github.com/crisie/RecurrentGaze/blob/master/generate_training_FT.py
Revised by Mehdi.
'''

EYEDIAP_path = "/data/eyediap"
import os
import cv2
import sys
import numpy as np
from tqdm import tqdm
from data_utils import *
from eyediap_utils import *
sys.path.append(os.path.join(EYEDIAP_path, 'Scripts'))
from data_utils import get_normalized_data

info_dict = {'fn': [], 'gaze': []}
num_saved_data = 0 

def getSessionIndex(P, C, T, H):
    sessions = []
    for P in range(16):
        if P < 11:
            Cs = ['A', ]
        elif P < 13:
            Cs = ['B', ]
        else:
            Cs = ['A', 'B']
        for C in Cs:
            if (P < 11 or P > 12) and C == 'A':
                Ts = ['DS', 'CS', 'FT']
            else:
                Ts = ['FT']
            for T in Ts:
                for H in ['S', 'M']:
                    sessions.append((P+1, C, T, H))

    try:
        idx = sessions.index((P, C, T, H))
        return idx
    except ValueError:
        return None

def get_session_string(session_idx):
    sessions = []
    for P in range(16):
        if P < 11:
            Cs = ['A', ]
        elif P < 13:
            Cs = ['B', ]
        else:
            Cs = ['A', 'B']
        for C in Cs:
            if (P < 11 or P > 12) and C == 'A':
                Ts = ['DS', 'CS', 'FT']
            else:
                Ts = ['FT']
            for T in Ts:
                for H in ['S', 'M']:
                    sessions.append((P+1, C, T, H))


    """
    Gets the string identifier for session "session_idx". It's useful for browsing the data
    """
    if session_idx >= 0 and session_idx < len(sessions):
        P, C, T, H = sessions[session_idx]
        return str(P)+'_'+C+'_'+T+'_'+H
    else:
        return ''

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


mod = 'FT'   # target type
type = ['S', 'M']  # type of head movement


# world coordinates to camera coordinates (see EYEDIAP for details)
Rw = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
tw = [0, 0, 1]
Rw = np.array(Rw)
tw = np.array(tw).reshape((3, 1))

for t in type:
    # Session selection (from EYEDIAP scripts)
    sessions = []
    for P in range(1, 17):
        if P < 12:
            Cs = ['A']
        elif P < 14:
            Cs = ['B']
        else:
            Cs = ['A', 'B']
        for C in Cs:
            sessions.append(getSessionIndex(P, C, mod, t))

    frames_done = 0
    seq_num_t = 0
    EYEDIAP_path = "/data/eyediap"
    for session_num in sessions:
        print("frames_done", frames_done)
        print("Session num: ", session_num)
        session_str = get_session_string(session_num)
        print("Session string: ", session_str)

        # EYEDIAP data path
        head_track_file = os.path.join(EYEDIAP_path, 'Data', session_str, 'head_pose.txt')
        ball_track_file = os.path.join(EYEDIAP_path, 'Data', session_str, 'ball_tracking.txt')
        valid_file = os.path.join(EYEDIAP_path, 'Annotations', 'GazeState', 'GazeStateExport', 'Data',
                                  session_str,
                                  'gaze_state.txt')
        eyeball_centers_file = os.path.join(EYEDIAP_path, 'Metadata', 'Participants',
                                            'eyes_position_%d' % int(session_str.split('_')[0]) + '.txt')
        vga_calibration = os.path.join(EYEDIAP_path, 'Data', session_str, 'rgb_vga_calibration.txt')

        # Video frames data path
        video_path = os.path.join(EYEDIAP_path, 'Data', session_str, 'rgb_vga.mov')
        frames = get_frames(video_path)

        # If annotations on valid frames are available, then these are loaded and taken into account
        valid_frames = None
        if os.path.exists(valid_file):
            valid_frames = np.genfromtxt(valid_file, delimiter='\t', dtype=None)


        # Read the files with the frame-by-frame tracking parameters
        landmarks = np.load(os.path.join(EYEDIAP_path, 'landmarks_3d', session_str, 'landmakrs_vga.npy'), allow_pickle=True)

        # Read the files with the frame-by-frame tracking parameters
        ball_track = read_ball_track_file(ball_track_file)
        head_track = read_head_track_file(head_track_file)

        # Read calibration file
        calib = readCalibrationFile(vga_calibration)

        # The estimated eyeball centers with respect to the head coordinate system
        eyeball_centers = np.loadtxt(eyeball_centers_file)

        # Containers for valid frames
        valid_head = np.zeros(ball_track.shape[0], dtype=np.bool)
        valid_ball = np.zeros(ball_track.shape[0], dtype=np.bool)
        valid_gaze = np.zeros(ball_track.shape[0], dtype=np.bool)
        valid_face = np.zeros(ball_track.shape[0], dtype=np.bool)
        valid_angles = np.zeros(ball_track.shape[0], dtype=np.bool)

        valid = False
        start_sequence = 0
        end_sequence = 0
        seq_num = 0
        eyeball_centers_moved = eyeball_centers.astype(dtype=np.float32, copy=True)
        eyeball_centers_moved[:, 1] = eyeball_centers_moved[:, 1] + 0.005
        for frameIndex in range(len(head_track[0])):
            print("...................")
            print("Frame: ", frameIndex)

            # Read the parameters for the current frame
            ball_pos = ball_track[frameIndex, :].reshape(3, 1)
            R = head_track[0][frameIndex, :, :]
            T = head_track[1][frameIndex, :].reshape(3, 1)

            # Validity criteria
            valid_head[frameIndex] = np.sum(T) != 0.0
            valid_ball[frameIndex] = np.sum(ball_pos) != 0
            valid_gaze[frameIndex] = valid_frames[frameIndex][1] == (b'OK' or b'BK') # Valid if OK or BlinKing
            valid_face[frameIndex] = np.sum(landmarks[frameIndex]) > 0.0

            # WCS to HCS system
            # Refer the ball center to the head coordinate system
            ball_HCS = np.dot(R.transpose(), ball_pos) - np.dot(R.transpose(), T)
            # Generate the ground truth gaze vectors
            gaze_vectors = ball_HCS.reshape(1, 3) - eyeball_centers
            gaze_vectors = gaze_vectors / (np.sqrt(np.sum(gaze_vectors ** 2, axis=1)).reshape(-1, 1))
            gaze_vector = np.mean(gaze_vectors, axis=0)  # we use the mean of the gaze vectors
            gaze_vector = gaze_vector / np.sqrt(np.sum(gaze_vector ** 2))
            gaze_angles = vector2angles(gaze_vector)

            # Valid angles according to (approx) max field of view when fixating in object
            valid_angles[frameIndex] = (-40 * np.pi / 180 < gaze_angles[0][0] < 40 * np.pi / 180) \
                           and (-30 * np.pi / 180 < gaze_angles[0][1] < 30 * np.pi / 180)

            valid_before = valid

            valid = valid_head[frameIndex] and valid_ball[frameIndex] and valid_gaze[frameIndex] \
                    and valid_face[frameIndex] and valid_angles[frameIndex]
            print("Valid: ", valid)
            validity_row = np.array((valid_head[frameIndex], valid_ball[frameIndex], valid_gaze[frameIndex],
                                     valid_angles[frameIndex], valid_face[frameIndex]))

            if not valid:
                if valid_before:
                    end_sequence = frameIndex - 1
                    seq_num = seq_num + 1
                    seq_num_t = seq_num_t + 1
                continue

            if not valid_before:
                start_sequence = frameIndex

            # WCS to CCS system
            eyeball_centers_WCS = np.dot(R, eyeball_centers.transpose()) + T
            eyeball_centers_moved_WCS = np.dot(R, eyeball_centers_moved.transpose()) + T

            eyeball_centers_CCS = np.dot(Rw.transpose(), eyeball_centers_WCS) - np.dot(Rw.transpose(), tw)
            eyeball_centers_moved_CCS = np.dot(Rw.transpose(), eyeball_centers_moved_WCS) - np.dot(Rw.transpose(), tw)

            ball_CCS = np.dot(Rw.transpose(), ball_pos) - np.dot(Rw.transpose(), tw)

            gaze_vectors_CCS = ball_CCS.reshape(1, 3) - eyeball_centers_CCS.transpose()

            gaze_vectors_CCS = gaze_vectors_CCS / (np.sqrt(np.sum(gaze_vectors_CCS ** 2, axis=1)).reshape(-1, 1))

            gaze_vector_CCS = np.mean(gaze_vectors_CCS, axis=0)

            gaze_vector_CCS = gaze_vector_CCS / np.sqrt(np.sum(gaze_vector_CCS ** 2))

            gaze_angles_CCS = vector2angles(gaze_vector_CCS)

            # Head direction in CCS
            aux_vector = np.zeros((3, 1), dtype=np.float32)
            aux_vector[2, 0] = 1.0
            head_vector_WCS = np.dot(R, aux_vector)
            head_vector_CCS = np.dot(Rw.transpose(), head_vector_WCS)
            head_vector_CCS = head_vector_CCS / np.sqrt(np.sum(head_vector_CCS ** 2))
            head_angle_CCS = vector2angles(head_vector_CCS)

            # Face normalization
            # we assume that center of face is 10 cm away from the center of the face in Z
            head = np.array([[0.0], [0.0], [0.1]]).reshape(1, 3)
            head_center_WCS = np.dot(head, R.transpose()) + T.transpose()

            head_center_CCS = np.dot(Rw.transpose(), head_center_WCS.transpose()) - np.dot(Rw.transpose(), tw)

            mean_face = head_center_CCS * 1000
            mean_face = mean_face.transpose()

            face_roi_size = [224, 224]
            face_patch_conv, face_patch_warp, face_patch_gaze = get_normalized_data(mean_face[0], R, face_roi_size,
                                                                                    calib)

            normalized_face = warp_image(frames[frameIndex], face_patch_warp, (224, 224))

            # both gaze angle and head angle are yaw, pitch so we reverse them to be as [pitch, yaw]
            gaze_angles_CCS = gaze_angles_CCS[:, ::-1]
            head_center_CCS = head_center_CCS[:, ::-1]

            cv2.imwrite(EYEDIAP_path + '/normalized_face_224_ft/' + '{}.png'.format(num_saved_data), normalized_face[:, :, ::-1])
            info_dict['fn'].extend([num_saved_data])
            info_dict['gaze'].append(gaze_angles_CCS)

            num_saved_data += 1
            print('Num Data: {}'.format(num_saved_data))
    


np.save(EYEDIAP_path + "/eyediap_ft", info_dict)



