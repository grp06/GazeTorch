'''
Converts directory of .lmk files to a single .lmk in list format.
* Need to initialize empty list of size 1008000 (total image ids) for indexing.
* Actual size is closer to ~940000 after processing.

'''
import os
import numpy as np
from tqdm import tqdm

PATH_ID = '/store/GazeGene/id_list_gazegene.npy'
PATH_LMK = '/home/joelee217/lmk_224'
PATH_CHECK = "/data/gazecapture_3d/lmk.npy"

def lmk_dir_to_npy():
    id_list = np.load(PATH_ID, allow_pickle=True)

    # init empty npy file
    glist = [None] * 1008000 # max idx/size of ids

    print(len(glist))

    # add each lmk to corresponding id in order following id_list
    for id in tqdm(id_list):
        filepath = f"{PATH_LMK}/{id}.npy"
        if os.path.isfile(filepath):
            lmk = list(np.load(filepath, allow_pickle=True).item().values())
            glist[id] = lmk

    # Save to .npy file
    np.save('/home/joelee217/lmk.npy', np.array(glist, dtype=object))

    print("saved to /home/joelee217/lmk.npy")

if __name__ == '__main__':
    lmk_dir_to_npy()
