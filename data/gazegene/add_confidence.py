import numpy as np
import os
from tqdm import tqdm
import cv2
# import sys
# sys.path.append("../..")

from cv.segmap import Segmap

ROOT = "/store/GazeGene"
fn_in = "gaze_gen_info.npy"
fn_target = "gazegene_gt_conf.npy"
id_list_path = "/store/GazeGene/id_list_gazegene.npy"
path_in = os.path.join(ROOT, fn_in)
path_target = os.path.join(ROOT, fn_target)

s = Segmap()

def batch():
    D = np.load(path_in, allow_pickle=True).item()
    id_list = np.load(id_list_path, allow_pickle=True)
    D['wE_L'] = []
    D['wI_L'] = []
    D['wE_R'] = []
    D['wI_R'] = []

    fns = D['fn']

    n_skipped = 0
    for i in tqdm(range(len(fns))):
        fn = fns[i]
        # id_img = str(id_list[i])
        path_face = os.path.join(ROOT, 'face_224', f"{i}.png")
        path_seg = os.path.join(ROOT, "seg", f'{i}.png')
        if not os.path.isfile(path_seg):
            confs = [0, 0, 0, 0]
            n_skipped += 1
        else:
            face = cv2.imread(path_face)[:,:,::-1]
            segmap = cv2.imread(path_seg, cv2.IMREAD_UNCHANGED)

            hp = D['headpose'][i]
            confs = s.get_confidence(face, segmap, hp)

        D['wE_L'].append(confs[0])
        D['wI_L'].append(confs[1])
        D['wE_R'].append(confs[2])
        D['wI_R'].append(confs[3])

    np.save(path_target, D)
    print("Voila. {} without segmaps.".format(n_skipped))

batch()