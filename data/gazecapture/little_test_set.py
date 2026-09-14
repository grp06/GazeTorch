import numpy as np
import os

from data.dataset.gc3d import Dataset_GC3d as GC3D

# little: 220 people, 10 samples each.
# 10-people: 10 people, 100 samples each.

root = '/data/gazecapture_3d'
source = 'gt_p17y17_with_conf_v2.npy'

def set1():
    target = 'gt_20p200s.npy'
    path_source = os.path.join(root, source)
    path_target = os.path.join(root, target)

    subjects = [
        # '01718', # problematic
        '00914',
        # '01057', # problematic
        '00208',
        '00733',
        '01569',
        '01199',
        '02998',
        '02255',
        '01566',
        '00097',
        '00492',
        '00645',
        '00613',
        '00889',
        '02236',
        '03413',
        '01445',
        '02368',
        '00872',
        '00658',
        '01316',
    ]

    np.random.seed(0)
    population = 200
    dataset = GC3D(inputs=['face', 'facemap'], labels=['gaze'], split='val', augment=False, 
        population=population, force_sample=True, subjects=subjects, sampler_do_shuffle=False)
    id_list = dataset.id_list
    print("# of id_list:", len(id_list))

    # np.save(os.path.join(root, 'little_idlist.npy'), id_list)

    D = np.load(path_source, allow_pickle=True).item()
    keys = list(D.keys())

    print("Keys:")
    D_little = {}
    for key in D:
        print(key)
        t = D[key]
        if not isinstance(t, list):
            D_little[key] = t
            continue
        D_little[key] = []
        for i in id_list:
            D_little[key].append(t[i])

    # print(D_little['subject_cid'])
    # D_little['subject_id_list'] = subjects

    fn = D_little['fn']
    print("Subjects:")
    for i in range(0, len(fn), population):
        print(fn[i][:5])

    np.save(path_target, D_little)
    print("Voila.")

def set2():
    '''
    "Chocolate box" - outputs 2 images per gaze bin
    '''
    import cv2
    from tqdm import tqdm
    target_path = "temp_results/gc3d_choco"
    n_per_grid = 2
    d = GC3D(inputs=['face', 'facemap'], labels=['gaze', 'bin', 'fn'], split='train', augment=False,
        augmenter_cfg_path="config/augmenter/identity.yaml", img_channel_chw=False)

    np.random.seed(0)
    ids = np.random.permutation(len(d))

    chocolate_box = np.zeros((d.gt_data['pitch_nbins'], d.gt_data['yaw_nbins']))
    n_grids = chocolate_box.shape[0] * chocolate_box.shape[1]
    for i in tqdm(ids):
        Xs, labels = d[i]
        face = Xs[0]*255
        face = face[:,:,::-1]

        bin = labels[1]
        p = bin[0]
        y = bin[1]
        fn = labels[2]
        if chocolate_box[p,y] < n_per_grid:
            # save an image
            target_fn = f"{target_path}/{p}_{y}_{fn}.png"
            cv2.imwrite(target_fn, face)
            chocolate_box[p,y] += 1
            if np.sum(chocolate_box) >= n_grids * n_per_grid:
                break

    print("Voila.")

def gt_from_fnlist(npy_list, npy_target):
    path_fnlist = os.path.join(root, npy_list)
    path_target = os.path.join(root, npy_target)
    fns = np.load(path_fnlist, allow_pickle=True)

    path_source = os.path.join(root, source)
    D = np.load(path_source, allow_pickle=True).item()
    fns_full = D['fn']

    D_little = {}
    for key in D:
        D_little[key] = []
    id_list = []

    for i, fn in enumerate(fns_full):
        if fn not in fns:
            continue
        id_list.append(i)
    
    print("# of samples:", len(id_list))

    D_little = {}
    for key in D:
        print(key)
        t = D[key]
        if not isinstance(t, list):
            D_little[key] = t
            continue
        D_little[key] = []
        for i in id_list:
            D_little[key].append(t[i])

    np.save(path_target, D_little)

gt_from_fnlist("choco_fns.npy", "choco.npy")