import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import joblib
import copy
from scipy import interpolate


dir_root = '/data/xgaze/xgaze_224/'
dir_train = dir_root + 'train/'
dir_test = dir_root + 'test/'

file_list = glob.glob(dir_root+'/train/*.h5') # 80 files
path_gt = '/data/xgaze/gt.npy'

'''
===[Dict.key list]===
'id' ~ same as "fn" in GC3D
'side'
---directly from H5---
'gaze'
'headpose'
'cam_index'
'frame_index'
'subject'
---processed---
pitch_cid
yaw_cid
cam_cid
subject_cid
pitch_nbins
yaw_nbins
num_L (single eye only)
num_R (single eye only)
'w_L'
'w_R'
'wE_L'
'wI_L'
'wE_R'
'wI_R'
'''

NUM_DATA=756540
START_VAL=672228
np.random.seed(0)

def load_gt():
    return np.load(path_gt, allow_pickle=True).item()

def generate_single_eye_gt(D0):
    # generate the classification labels
    n_bins_pitch = 17
    n_bins_yaw = 17
    bin_size = 4.

    pitch_limit = (n_bins_pitch//2+0.5) * bin_size # 62
    yaw_limit = (n_bins_yaw//2+0.5) * bin_size # 90

    map_cam_L = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}
    map_cam_R = {0:1, 1:0, 6:2, 5:3, 4:4, 3:5}

    D = {}
    D['id'] = [] # id in entire dataset (0~NUM_DATA-1)
    D['side'] = [] # 'L' or 'R'
    D['gaze'] = []
    D['headpose'] = []
    D['cam_index'] = []
    D['frame_index'] = []
    D['subject'] = []
    D['num_L'] = 0
    D['num_R'] = 0
    D['pitch_cid'] = []
    D['yaw_cid'] = []
    D['cam_cid'] = []
    # and we have D['num_L'] and D['num_R']
    D_val = copy.deepcopy(D)

    M = 756540

    root = '/data/xgaze/448'
    for dn in ['eye_L_aug_0415', 'eye_R_aug_0415']:
        c = dn[4] # 'L' or 'R'
        key_num = 'num_'+c
        map_cam = eval("map_cam_"+c)
        for i in tqdm(range(M)):
            if not i%18 in map_cam.keys(): continue
            path = os.path.join(root, dn, "%d.jpg"%i)
            if not os.path.isfile(path): continue
            # Dc = D if i<START_VAL else D_val
            Dc = D
            for key in ['gaze','headpose','cam_index','frame_index','subject']:
                Dc[key].append(D0[key][i])
            Dc['id'].append(i)
            Dc['side'].append(c) 
            Dc[key_num] += 1

            gaze = D0['gaze'][i]
            # cid = id label in classification problem
            pitch_cid = int(np.clip((gaze[0] * 180. / np.pi + pitch_limit) // bin_size, 0, n_bins_pitch-1))
            yaw_cid = int(np.clip((gaze[1] * 180. / np.pi + yaw_limit) // bin_size, 0, n_bins_yaw-1))

            cam_id = D0['cam_index'][i][0] - 1
            cam_cid = map_cam[cam_id]

            Dc['pitch_cid'].append(pitch_cid)
            Dc['yaw_cid'].append(yaw_cid)
            Dc['cam_cid'].append(cam_cid)
    D['pitch_nbins'] = n_bins_pitch
    D['yaw_nbins'] = n_bins_yaw
    # D_val['pitch_nbins'] = n_bins_pitch
    # D_val['yaw_nbins'] = n_bins_yaw
    print("D:", len(D['id']))
    # print("D_val:", len(D_val['id']))
    # np.save(os.path.join(root, 'single_eye_train_p17y17.npy'), D)
    # np.save(os.path.join(root, 'single_eye_val_p17y17.npy'), D_val)
    np.save(os.path.join(root, 'single_eye_p17y17.npy'), D)
    print("Voila.")

def generate_double_eye_gt(D0):
    cams = (0,1,3,4,5,9)
    map_cam = {0:0, 1:1, 3:2, 4:3, 5:4, 9:5}

    n_bins_pitch = 17
    n_bins_yaw = 17
    bin_size = 4.
    pitch_bias = -16

    pitch_limit = (n_bins_pitch//2+0.5) * bin_size # 62
    yaw_limit = (n_bins_yaw//2+0.5) * bin_size # 90

    D = {}
    D['id'] = [] # id in entire dataset (0~NUM_DATA-1)
    # D['side'] = [] # 'L' or 'R'
    D['gaze'] = []
    D['headpose'] = []
    D['cam_index'] = []
    D['frame_index'] = []
    D['subject'] = []
    D['cam_cid'] = []
    D['pitch_cid'] = []
    D['yaw_cid'] = []
    # and we have D['num_L'] and D['num_R']
    D_val = copy.deepcopy(D)

    M = 756540
    dns = ['eye_L_0610', 'eye_R_0610']

    root = '/data/xgaze/448'
    for i in tqdm(range(M)):
        if not i%18 in cams: continue
        path_L = os.path.join(root, dns[0], "%d.jpg"%i)
        path_R = os.path.join(root, dns[1], "%d.jpg"%i)
        if (not os.path.isfile(path_L)) or (not os.path.isfile(path_R)): continue
        Dc = D if i<START_VAL else D_val
        for key in ['gaze','headpose','cam_index','frame_index','subject']:
            Dc[key].append(D0[key][i])
        Dc['id'].append(i)
        Dc['cam_cid'].append(map_cam[D0['cam_index'][i][0]-1])

        gaze = D0['gaze'][i]
        # cid = id label in classification problem
        pitch_cid = int(np.clip((gaze[0] * 180. / np.pi + pitch_limit - pitch_bias) // bin_size, 0, n_bins_pitch-1))
        yaw_cid = int(np.clip((gaze[1] * 180. / np.pi + yaw_limit) // bin_size, 0, n_bins_yaw-1))
        Dc['pitch_cid'].append(pitch_cid)
        Dc['yaw_cid'].append(yaw_cid)

    for Dc in (D, D_val):
        Dc['pitch_nbins'] = n_bins_pitch
        Dc['yaw_nbins'] = n_bins_yaw
        Dc['pitch_bias'] = pitch_bias
        Dc['bin_size'] = bin_size
        
    print("D:", len(D['id']))
    print("D_val:", len(D_val['id']))
    np.save(os.path.join(root, 'double_eye_train_0610p17y17b16.npy'), D)
    np.save(os.path.join(root, 'double_eye_val_0610p17y17b16.npy'), D_val)
    print("Voila.")

def get_confidence(yaw_in_degrees, eye='L'):
    if eye == 'L':
        if yaw_in_degrees < 0:
            return 1
        elif yaw_in_degrees < 60:
            points = np.array([[0, 1], [10, 0.95], [20, 0.8], [30, 0.6], [50, 0.1],[60, 0]])
            x = points[:, 0]
            y = points[:, 1]
            func = interpolate.splrep(x, y, s=0)
            conf = float(interpolate.splev(yaw_in_degrees, func, der=0))
            if conf > 1:
                conf = 1
            return conf
        else:
            return 0

    else:
        if yaw_in_degrees < -60:
            return 0
        elif yaw_in_degrees < 0:
            points = np.array([[-60, 0], [-50, 0.1], [-30, 0.6], [-20, 0.8], [-10, 0.95], [0, 1]])
            x = points[:, 0]
            y = points[:, 1]
            func = interpolate.splrep(x, y, s=0)
            conf = float(interpolate.splev(yaw_in_degrees, func, der=0))
            if conf > 1:
                conf = 1
            return conf
        else:
            return 1

def extract_feature(face, facemap, headpose):
    leye = facemap[:, :, 0]
    liris = facemap[:, :, 1]
    reye = facemap[:, :, 2]
    riris = facemap[:, :, 3]
    pitch = headpose[0] * 180 / np.pi
    yaw = headpose[1] * 180 / np.pi
    leye[leye != 0] = 255
    reye[reye != 0] = 255
    liris[liris != 0] = 255
    riris[riris != 0] = 255
    leye_pix = np.count_nonzero(leye == 255)
    reye_pix = np.count_nonzero(reye == 255)
    liris_pix = np.count_nonzero(liris == 255)
    riris_pix = np.count_nonzero(riris == 255)
    mean_illum = np.mean(face)
    feature = [leye_pix, liris_pix, reye_pix, riris_pix] + [mean_illum] + [pitch, yaw]

    return feature


def generate_all_gt(D0):
    points = np.array([[0, 1], [10, 0.95], [20, 0.8], [30, 0.6], [50, 0.1],[60, 0]])
    x = points[:, 0]
    y = points[:, 1]
    conf_left_func = interpolate.splrep(x, y, s=0)

    confidence_regressor = joblib.load('cv/modules/confidence_regressor.sav')
    confidence_scaler = joblib.load('cv/modules/confidence_scaler.sav')

    n_bins_pitch = 17
    n_bins_yaw = 17
    bin_size = 4.
    pitch_bias = -16

    pitch_limit = (n_bins_pitch//2+0.5) * bin_size # 62
    yaw_limit = (n_bins_yaw//2+0.5) * bin_size # 90

    D = {}
    D['id'] = [] # id in entire dataset (0~NUM_DATA-1)
    D['gaze'] = []
    D['headpose'] = []
    D['cam_index'] = []
    D['frame_index'] = []
    D['subject'] = []
    D['cam_cid'] = []
    D['pitch_cid'] = []
    D['yaw_cid'] = []
    D['eye_L_avail'] = [] # if left eye image is available
    D['eye_R_avail'] = [] # if right eye image is available
    D['w_L'] = []
    D['w_R'] = []
    D['wE_L'] = []
    D['wI_L'] = []
    D['wE_R'] = []
    D['wI_R'] = []

    M = 756540
    dns = ['eye_L_0610', 'eye_R_0610']

    root = '/data/xgaze/448'
    n_skipped = 0
    for i in tqdm(range(M)):
        path_L = os.path.join(root, dns[0], "%d.jpg"%i)
        path_R = os.path.join(root, dns[1], "%d.jpg"%i)

        D['eye_L_avail'] = os.path.isfile(path_L)
        D['eye_R_avail'] = os.path.isfile(path_R)

        for key in ['gaze','headpose','cam_index','frame_index','subject']:
            D[key].append(D0[key][i])

        D['w_L'].append(get_confidence(D0['headpose'][i][1]*180/np.pi, eye='L'))
        D['w_R'].append(get_confidence(D0['headpose'][i][1]*180/np.pi, eye='R'))

        face_dir = root + '/face_224/' + str(i) + '.jpg'
        facemap_dir = face_dir.replace('.jpg', '.png').replace('face_224', 'facemap_224')
        face = cv2.imread(face_dir)
        facemap = cv2.imread(facemap_dir, cv2.IMREAD_UNCHANGED)
        headpose = D0['headpose'][i]
        if facemap is None:
            confs = [0, 0, 0, 0]
            n_skipped += 1
        else:
            feature = extract_feature(face, facemap, headpose)
            feature = np.array(feature).reshape([1, -1])
            feature_n = confidence_scaler.transform(feature)
            confs = confidence_regressor.predict(feature_n)[0]
            confs[confs < 0.25] = 0
            confs[confs >= 0.9] = 1
            confs[(np.where((confs >= 0.25) & (confs <= 0.9)))] = 1.5385 * (confs[(np.where((confs >= 0.25) & (confs <= 0.9)))] - 0.25)
            confs[confs >= 1] = 1
            confs[confs <= 0] = 0

        D['wE_L'].append(confs[0]) 
        D['wI_L'].append(confs[1])
        D['wE_R'].append(confs[2])
        D['wI_R'].append(confs[3])


        D['id'].append(i)
        D['cam_cid'].append(D0['cam_index'][i][0]-1)
        gaze = D0['gaze'][i]
        # cid = id label in classification problem
        pitch_cid = int(np.clip((gaze[0] * 180. / np.pi + pitch_limit - pitch_bias) // bin_size, 0, n_bins_pitch-1))
        yaw_cid = int(np.clip((gaze[1] * 180. / np.pi + yaw_limit) // bin_size, 0, n_bins_yaw-1))
        D['pitch_cid'].append(pitch_cid)
        D['yaw_cid'].append(yaw_cid)
  
    D['pitch_nbins'] = n_bins_pitch
    D['yaw_nbins'] = n_bins_yaw
    D['pitch_bias'] = pitch_bias
    D['bin_size'] = bin_size
        
    print("# of data in D:", len(D['id']))
    np.save(os.path.join(root, 'gt_p17y17b16_with_confs_v2.npy'), D)
    print('Number of skipped confidences: {}'.format(n_skipped))
    print("Voila.")

def generate_population():
    import h5py
    root_h5 = '/data/xgaze/xgaze_448'
    D = {}
    D['id'] = []
    D['count'] = []
    D['start_id'] = []
    counter = 0
    for fn in tqdm(os.listdir(os.path.join(root_h5, "train"))):
        path_h5 = os.path.join(root_h5, "train", fn)
        D['id'].append(fn[:-3])
        h5 = h5py.File(path_h5)
        count = len(h5['face_gaze'])
        D['count'].append(count)
        D['start_id'].append(counter)
        counter += count
    np.save("/data/xgaze/448/population.npy", D)
    print("Voila.")

def add_subject(path_npy):
    path_pop = "/data/xgaze/448/population.npy"
    D_pop = np.load(path_pop, allow_pickle=True).item()
    subject_id_list = [0 for _ in range(NUM_DATA)]
    for i, (start_id, count) in enumerate(zip(D_pop['start_id'], D_pop['count'])):
        print(i, start_id, count)
        for j in range(start_id, start_id+count):
            subject_id_list[j] = i
    
    D = np.load(path_npy, allow_pickle=True).item()
    D['subject_cid'] = subject_id_list
    np.save(path_npy, D)
    print("Voila.")

def fix_key_name_weightv2(path_npy):
    '''
    A temp function to change key names.
    '''
    D = np.load(path_npy, allow_pickle=True).item()
    old_keys = ("wL_E_v2", "wL_I_v2", "wR_E_v2", "wR_I_v2")
    new_keys = ("wE_L", "wI_L", "wE_R", "wI_R")
    for k_o, k_n in zip(old_keys, new_keys):
        D[k_n] = D.pop(k_o)
    np.save(path_npy, D)
    print("Voila.")

if __name__=="__main__":
    D0=load_gt()
    generate_all_gt(D0)
    # generate_list_split(file_list)

    # D0=load_gt()
    # generate_single_eye_gt(D0)
    # generate_double_eye_gt(D0)
    # generate_population()
    add_subject('/data/xgaze/448/gt_p17y17b16_with_confs_v2.npy')
    #fix_key_name_weightv2('/data/xgaze/448/gt_p17y17b16_with_confs_v2.npy')