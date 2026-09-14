import numpy as np
import os
from tqdm import tqdm
from scipy import interpolate
import joblib
import cv2


root = '/data/rt_gene'

'''
'w_L'
'w_R'
'wE_L'
'wI_L'
'wE_R'
'wI_R'
'headpose'
'w_wrt_subject_id'
'''


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
  
    n_skipped = 0
    for i in tqdm(range(len(D['headpose']))):
        face_dir = root + '/face_224/' + str(D['fn'][i]) + '.png'
        facemap_dir = face_dir.replace('face_224', 'facemap')
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


def add_weight_wrt_subject_id(filename_in, filename_out):
    path_in= os.path.join(root, filename_in)
    path_out= os.path.join(root, filename_out)
    D = np.load(path_in, allow_pickle=True).item()
    subjects = D['subject_cid']
    unique, counts = np.unique(D['subject_cid'], return_counts=True)
    dict_count = dict(zip(unique, counts))
    w_subjects = []
    for i in range(len(D['fn'])):
        subject = D['subject_cid'][i]
        num_data_subject = dict_count[subject]
        w_subjects.extend([1./num_data_subject])
    
    D['w_wrt_subject_id'] = w_subjects
    np.save(path_out, D)
    print("Voila.")
  
if __name__=='__main__':
    add_eye_confidence_v2('rt_gene.npy', 'gt_rt_gene_with_conf_v2.npy')
    add_weight_wrt_subject_id('gt_rt_gene_with_conf_v2.npy', 'gt_rt_gene_with_conf_v2.npy')