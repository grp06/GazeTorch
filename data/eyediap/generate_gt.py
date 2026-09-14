import os
import numpy as np
import eyediap_utils
from tqdm import tqdm

ROOT = '/data/eyediap'

def generate_gaze3d():
    # load ROOT/Data to fetch 3d gaze GT
    # generate the original GT file too (id - image path - gt)
    root_data = os.path.join(ROOT, 'Data')
    root_target = os.path.join(ROOT, 'gaze_3d_angles')

    sess_names = sorted(os.listdir(root_data))
    for sess in sess_names:
        print(sess)
        path_coord = os.path.join(root_data, sess, 'screen_coordinates.txt')
        path_eye = os.path.join(root_data, sess, 'eye_tracking.txt')
        path_out_g3d = os.path.join(root_target, sess + '.txt')

        with open(path_coord) as f:
            lines_coord = f.readlines()
        with open(path_eye) as f:
            lines_eye = f.readlines()
        lines_g3d = []
        for i in range(16, len(lines_eye), 15):
            line_coord = lines_coord[i]
            g_raw = [float(word) for word in line_coord.strip().split(';')]
            frame_id = g_raw[0]
            g_x, g_y, g_z = g_raw[3:6]
            line_eye = lines_eye[i]
            eye_vals = [float(word) for word in line_eye.strip().split(';')]
            eye_x, eye_y, eye_z = [eye_vals[-3], eye_vals[-2], eye_vals[-1]]
            gv_x, gv_y, gv_z = [g_x - eye_x, g_y - eye_y, g_z - eye_z]
            gv = np.asarray([gv_x, gv_y, gv_z])
            gv = gv.reshape(3, 1)
            v1, v2 = eyediap_utils.vector2angles(gv)
            line_g3d = "%d;%f;%f"%(frame_id, v1, v2)
            lines_g3d.append(line_g3d)
        
        with open(path_out_g3d, 'w') as f:
            f.write('\n'.join(lines_g3d))
    print("Voila.")

generate_gaze3d()