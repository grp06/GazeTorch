"""
xgaze: select from cam_id 0, 1, 2, 3, 4, 5, 6
alex2: don't want images with face covering -> gt mask == 0
"""

import numpy as np
import random
import os

DATASETS = ['alex2', 'eyediap', 'gaze360', 'gc3d', 'mpiiface', 'xgaze']

ROOTS = {
    'alex2': ["/data/lzh/alex2_full", "/data/lzh/alex2_out", "/data/weiye/dataR"],
    'eyediap': "/data/eyediap",
    'gaze360': '/data/gaze360',
    'gc3d': '/data/gazecapture_3d',
    'mpiiface': "/data/mpiiface",
    'xgaze': "/data/xgaze/448"
}

GTS = {
    'alex2': "gt_v1.npy",
    'eyediap': "gt_eyediap_cs_with_conf_v3.npy",
    'gaze360': "gt_gaze360_with_conf_v3.npy",
    'gc3d': "gt_p17y17_with_conf_v3.npy",
    'mpiiface': "gt_mpiiface_with_conf_v3.npy",
    'xgaze': "gt_p17y17b16_with_confs_v3.npy"
}

SEGS = {
    'alex2': "seg",
    'eyediap': 'seg_cs_bh',
    'gaze360': "seg",
    'gc3d': "seg",
    'mpiiface': "seg",
    'xgaze': "seg"
}

FACES = {
    'alex2': "face",
    'eyediap': 'normalized_face_224_cs_bh',
    'gaze360': "face_224",
    'gc3d': "face",
    'mpiiface': "224png",
    'xgaze': "face_224"
}

N = 200

XGAZE_CAM_ID = [0, 1, 2, 3, 4, 5, 6]


def get6n():
    face_dirs = []
    seg_dirs = []
    gaze = []

    # [!!!] not used for generating the first list of samples
    random.seed(0)

    for dataset in DATASETS:
        print("[Debug] On Dataset: ", dataset)
        # load ground truth
        if dataset != "alex2":
            # print("[Debug] Load GT Dataset Alex2")
            path_gt = os.path.join(ROOTS[dataset], GTS[dataset])
        else:
            # print("[Debug] Load GT not Alex2")
            path_gt = os.path.join(ROOTS[dataset][0], GTS[dataset])

        gt = np.load(path_gt, allow_pickle=True).item()

        num_total = len(gt['gaze'])

        # select N items from dataset
        i = 0
        while i < N:
            print("[Debug] Iteration: ", i)
            temp_index = random.randint(0, num_total-1)

            # for xgaze dataset, only takes images with camera ids in xgaze_cam_id list
            if dataset == "xgaze" and (gt['cam_index'][temp_index] not in XGAZE_CAM_ID):
                continue

            # for alex2 dataset, exclude images with mask
            if dataset =="alex2" and (gt["mask"][temp_index] == 1):
                continue

            if dataset == "xgaze":
                # print("[Debug] Load FN Dataset Xgaze")
                fn = str(gt['id'][temp_index])
            else:
                # print("[Debug] Load FN not Xgaze")
                fn = str(gt['fn'][temp_index])

            if dataset == "xgaze" or dataset == "gc3d":
                ext = ".jpg"
                ext_seg = ".png"
            else:
                ext = '.png'
                ext_seg = ".png"

            if dataset == "alex2":
                # fn = gt['fn'][temp_index]
                dn = gt['dn'][temp_index]
                dataset_id = gt['dataset_id'][temp_index]
                if dataset_id == 0:
                    print("[Debug] get face and seg path alex2 dataset id 0")
                    face_path = os.path.join(ROOTS[dataset][0], FACES[dataset], fn + ext)
                    seg_path = os.path.join(ROOTS[dataset][0], SEGS[dataset], fn + ext_seg)
                elif dataset_id == 1:
                    print("[Debug] get face and seg path alex2 dataset id 1")
                    face_path = os.path.join(ROOTS[dataset][1], FACES[dataset], fn + ext)
                    seg_path = os.path.join(ROOTS[dataset][1], SEGS[dataset], fn + ext_seg)
                else:
                    print("[Debug] get face and seg path alex2 dataset id 2")
                    face_path = os.path.join(ROOTS[dataset][2], dn, FACES[dataset], fn + ext)
                    seg_path = os.path.join(ROOTS[dataset][2], dn, SEGS[dataset], fn + ext_seg)
            else:
                print("[Debug] get face and seg path not alex2, on dataset: ", dataset)
                face_path = os.path.join(ROOTS[dataset], FACES[dataset], fn + ext)
                seg_path = os.path.join(ROOTS[dataset], SEGS[dataset], fn + ext_seg)

            # if both files exits
            if os.path.isfile(face_path) and os.path.isfile(seg_path):
                print("[Debug] File Exist")
                face_dirs.append(face_path)
                seg_dirs.append(seg_path)
                gaze.append(gt['gaze'][temp_index])
                i += 1

    return face_dirs, seg_dirs, gaze


face_dirs, seg_dirs, gaze = get6n()
print(face_dirs)
print(seg_dirs)
print(gaze)
print("Total nums: ", len(face_dirs), len(seg_dirs), len(gaze))
np.save("/data/ninajiang/select_6n/test_restriction_alex2_xgaze/alex_xgaze_face_dirs.npy", face_dirs)
np.save("/data/ninajiang/select_6n/test_restriction_alex2_xgaze/alex_xgaze_seg_dirs.npy", seg_dirs)
np.save("/data/ninajiang/select_6n/test_restriction_alex2_xgaze/alex_xgaze_gaze.npy", gaze)
