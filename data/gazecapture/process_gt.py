import numpy as np
import os
from tqdm import tqdm
from scipy import interpolate
import joblib
import cv2


root = '/data/gazecapture_3d'
PATH_POP = os.path.join(root, "population.npy")

'''
===[Dict.key list]===
'fn' - processed_entry['file_name'] ~ aaaaa_bbbbb
---GT---
'gaze' - processed_entry['normalized_gaze_direction'])
'gaze_L' - processed_entry['normalized_gaze_direction_left'])
'gaze_R' - processed_entry['normalized_gaze_direction_right'])
'w_L'
'w_R'
'wE_L'
'wI_L'
'wE_R'
'wI_R'
'headpose' # head_pose has been removed
'w_wrt_subject_id'
'subject_cid'
---GT quantized---
pitch_cid
yaw_cid
cam_cid
pitch_nbins
yaw_nbins
bin_size
pitch_bias
---IDs---
start_ids: Dict. first id (value) for each person (key)

===population.npy===
dict.
key = subject (in str, e.g. "00002")
value = population

===[train/val.npy key list]===
'id': list[str] lists Subject IDs
'polulation': list[int]

===roster.npy===
dict.
key = subject_id (in str, e.g. "00002")
value = subject_id (int, 0-base)
can be merged into population.npy actually
'''

def merge_gt():
    keys = ('fn', 'gaze', 'gaze_L', 'gaze_R', 'head_pose')
    D = {}
    for k in keys:
        D[k] = []

    for fn in sorted(os.listdir(os.path.join(root, 'gt'))):
        if not fn.endswith('.npy'): continue
        print(fn)
        npy = np.load(os.path.join(root, 'gt', fn), allow_pickle=True).item()
        for k in keys:
            D[k] += list(npy[k])
    
    D['num_data'] = len(D['fn'])
    np.save(os.path.join(root, 'gt_raw.npy'), D)
    print("Voila.")

def population():
    D = {}
    for fn in sorted(os.listdir(os.path.join(root, 'gt'))):
        if not fn.endswith('.npy'): continue
        print(fn)
        npy = np.load(os.path.join(root, 'gt', fn), allow_pickle=True).item()
        key = fn[:5]
        D[key] = len(npy['fn'])
    
    np.save(PATH_POP, D)
    print("Voila.")

def quantize(target_filename, n_bins_pitch=13, n_bins_yaw=17, bin_size=4., pitch_bias=-8.):
    pitch_limit = (n_bins_pitch//2+0.5) * bin_size 
    yaw_limit = (n_bins_yaw//2+0.5) * bin_size 
    path_raw = os.path.join(root, "gt_raw.npy")
    path_out = os.path.join(root, target_filename)
    D = np.load(path_raw, allow_pickle=True).item()
    D['pitch_cid'] = []
    D['yaw_cid'] = []
    D['start_ids'] = {}
    current_person_id = ""
    for i in tqdm(range(len(D['fn']))):
        gaze = D['gaze'][i]
        pitch_cid = int(np.clip((gaze[0] * 180. / np.pi + pitch_limit - pitch_bias) // bin_size, 0, n_bins_pitch-1))
        yaw_cid = int(np.clip((gaze[1] * 180. / np.pi + yaw_limit) // bin_size, 0, n_bins_yaw-1))
        D['pitch_cid'].append(pitch_cid)
        D['yaw_cid'].append(yaw_cid)

        person_id = D['fn'][i][:5]
        if person_id!=current_person_id:
            D['start_ids'][person_id] = i
            current_person_id = person_id
    D['pitch_nbins'] = n_bins_pitch
    D['yaw_nbins'] = n_bins_yaw
    D['pitch_bias'] = pitch_bias
    D['bin_size'] = bin_size
    np.save(path_out, D)
    print("Voila.")

def add_subject_id(filename, save_roster=True):
    path_npy= os.path.join(root, filename)
    D = np.load(path_npy, allow_pickle=True).item()
    D['subject_cid'] = []
    subjects = {}

    for key in ('fn', 'id'):
        if key in D:
            break

    counter = -1
    for i in tqdm(range(len(D[key]))):
        person_id = D[key][i][:5]
        if person_id not in subjects:
            counter += 1
            subjects[person_id] = counter
        D['subject_cid'].append(counter)
    np.save(path_npy, D)

    if save_roster:
        path_roster = os.path.join(root, "roster.npy")
        np.save(path_roster, subjects)

    print("Voila. # of subjects = ", counter+1)

def get_confidence(yaw_in_degrees, eye='L'):
    # confidence score V1 by Mehdi.
    # for left eye
    if eye == 'L':
        if yaw_in_degrees < 0:
            return 1
        elif yaw_in_degrees < 60:
            points = np.array([[0, 1], [10, 0.95], [20, 0.8], [30, 0.6], [50, 0.1],[60, 0]])
            x = points[:, 0]
            y = points[:, 1]
            func = interpolate.splrep(x, y, s=0)
            conf = float(interpolate.splev(yaw_in_degrees, func, der=0))
            # clip if the interpolation gives value more than 1
            if conf > 1:
                conf = 1
            return conf
        else:
            return 0
    # for right eye
    else:
        if yaw_in_degrees < -60:
            return 0
        elif yaw_in_degrees < 0:
            points = np.array([[-60, 0], [-50, 0.1], [-30, 0.6], [-20, 0.8], [-10, 0.95], [0, 1]])
            x = points[:, 0]
            y = points[:, 1]
            func = interpolate.splrep(x, y, s=0)
            conf = float(interpolate.splev(yaw_in_degrees, func, der=0))
            # clip if the interpolation gives value more than 1
            if conf > 1:
                conf = 1
            return conf
        else:
            return 1

def add_eye_confidence(load_filename, save_filename):
    load_path_npy= os.path.join(root, load_filename)
    D = np.load(load_path_npy, allow_pickle=True).item()
    D['w_L'] = []
    D['w_R'] = []

    # change head_pose to headpose
    if 'head_pose' in D.keys():
        D['headpose'] = D.pop('head_pose')

    for i in tqdm(range(len(D['headpose']))):
        D['w_L'].append(get_confidence(D['headpose'][i][1]*180/np.pi, eye='L'))
        D['w_R'].append(get_confidence(D['headpose'][i][1]*180/np.pi, eye='R'))

    save_path_npy= os.path.join(root, save_filename)
    np.save(save_path_npy, D)
    print("Voila.")

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

def add_eye_confidence_v2(load_filename, save_filename):
    load_path_npy= os.path.join(root, load_filename)
    D = np.load(load_path_npy, allow_pickle=True).item()
    D['wE_L'] = []
    D['wI_L'] = []
    D['wE_R'] = []
    D['wI_R'] = []
    confidence_regressor = joblib.load('cv/modules/confidence_regressor.sav')
    confidence_scaler = joblib.load('cv/modules/confidence_scaler.sav')

    # change head_pose to headpose
    if 'head_pose' in D.keys():
        D['headpose'] = D.pop('head_pose')

    n_skipped = 0
    for i in tqdm(range(len(D['headpose']))):
        face_dir = root + '/face_112/' + str(D['fn'][i]) + '.jpg'
        facemap_dir = face_dir.replace('.jpg', '.png').replace('face_112', 'facemap')
        face = cv2.imread(face_dir)
        face = cv2.resize(face, (224, 224))
        facemap = cv2.imread(facemap_dir, cv2.IMREAD_UNCHANGED)
        headpose = D['headpose'][i]      
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
            confs[(np.where((confs >= 0.25) & (confs <= 0.9)))] = 1.54 * (confs[(np.where((confs >= 0.25) & (confs <= 0.9)))] - 0.25)
            confs[confs >= 1] = 1
            confs[confs <= 0] = 0

        D['wE_L'].append(confs[0])
        D['wI_L'].append(confs[1])
        D['wE_R'].append(confs[2])
        D['wI_R'].append(confs[3])

    save_path_npy= os.path.join(root, save_filename)
    np.save(save_path_npy, D)
    print('Number of skipped confidences {} out of {}'.format(n_skipped, len(D['headpose'])))
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

# split dataset into train & val
def split(num_val = 250):
    D_pop = np.load(PATH_POP, allow_pickle=True).item()
    D_pop_sorted = {k:v for k,v in sorted(D_pop.items(), key=lambda item: item[1])} # only works for Python 3.7+

    D_train = {}
    D_val = {}
    keys = list(D_pop_sorted.keys())
    values = list(D_pop_sorted.values())
    
    D_val['id'] = keys[:num_val]
    D_val['population'] = values[:num_val]

    D_train['id'] = keys[num_val:]
    D_train['population'] = values[num_val:]

    np.save(os.path.join(root, "train250.npy"), D_train)
    np.save(os.path.join(root, "val250.npy"), D_val)

def add_weight_wrt_subject_id(filename_in, filename_out):
    path_in= os.path.join(root, filename_in)
    path_out= os.path.join(root, filename_out)

    D = np.load(path_in, allow_pickle=True).item()
    D_pop = np.load(PATH_POP, allow_pickle=True).item()

    weights = []
    for fn in tqdm(D['fn']):
        person_id = fn[:5]
        pop = D_pop[person_id]
        weights.append(1./pop)
    D['w_wrt_subject_id'] = weights
    np.save(path_out, D)
    print("Voila.")

if __name__=='__main__':
    # quantize('gt_p17y17.npy', 17, 17, 4., -8.)
    # split()
    #add_subject_id('gt_p17y17.npy')
    #add_subject_id('train250.npy')
    #add_subject_id('val250.npy')
    #add_eye_confidence('gt_p17y17.npy', 'gt_p17y17_with_conf.npy')
    add_eye_confidence_v2('gt_p17y17_with_conf.npy', 'gt_p17y17_with_conf_v2.npy')
    #fix_key_name_weightv2(os.path.join(root, 'gt_p17y17_with_conf_v2.npy'))

    add_weight_wrt_subject_id('gt_p17y17_with_conf_v2.npy', 'gt_p17y17_with_conf_v2.npy')