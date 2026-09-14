import numpy as np
from shutil import copy
import os

path = "/store/GazeGene/gazegene_gt_conf.npy"
root = "/store/GazeGene"

D = np.load(path, allow_pickle=True).item()
id_list_raw = np.load(os.path.join(root, "id_list_gazegene.npy"))

hp = np.array(D['headpose'])
hp_abs = np.abs(hp)

target_dir = 'temp/gazegene'

counter = 0
counter_exists = 0
valid_ids = []

for i in id_list_raw:
    path_face = os.path.join(root, "face_224", f"{i}.png")
    path_seg = os.path.join(root, "seg", f'{i}.png')
    if not os.path.isfile(path_face) or not os.path.isfile(path_seg):
        continue

    if hp_abs[i,0] < np.pi/6 and hp_abs[i,1] < np.pi/6 and hp_abs[i,2] < np.pi/36:
        valid_ids.append(i)

        # path_dst = f"{target_dir}/{i}.png"
        # copy(path_face, path_dst)
        # counter += 1
        # if counter == 100: break

    counter_exists += 1

np.save(os.path.join(root, "valid_ids_roll_pi36.npy"), valid_ids)

print(f"Voila. {len(valid_ids)} / {counter_exists}.")