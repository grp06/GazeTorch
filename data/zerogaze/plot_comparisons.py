import numpy as np
import os, sys
from tools.plotter import Plotter

PATH_VANILLA =f"/data/sfhq_fluxdev_gazezero_v2/eval/251128e35_gaze_zerogaze_vanilla.npy"
PATH_GLASSES =f"/data/sfhq_fluxdev_gazezero_v2/eval/251128e35_gaze_zerogaze_glasses.npy"
PATH_MASKS =f"/data/sfhq_fluxdev_gazezero_v2/eval/251128e35_gaze_zerogaze_masks.npy"

UH_PATH_VANILLA =f"/data/sfhq_fluxdev_gazezero_v2/eval/unigaze_gaze_zerogaze_vanilla.npy"
UH_PATH_GLASSES =f"/data/sfhq_fluxdev_gazezero_v2/eval/unigaze_gaze_zerogaze_glasses.npy"
UH_PATH_MASKS =f"/data/sfhq_fluxdev_gazezero_v2/eval/unigaze_gaze_zerogaze_masks.npy"

UB_PATH_VANILLA =f"/data/sfhq_fluxdev_gazezero_v2/eval/unigaze_B_gaze_zerogaze_vanilla.npy"
UB_PATH_GLASSES =f"/data/sfhq_fluxdev_gazezero_v2/eval/unigaze_B_gaze_zerogaze_glasses.npy"
UB_PATH_MASKS =f"/data/sfhq_fluxdev_gazezero_v2/eval/unigaze_B_gaze_zerogaze_masks.npy"

save_path = "/data/sfhq_fluxdev_gazezero_v2/eval/"

def load_data(path):
    data = np.load(path, allow_pickle=True)
    return data

vanilla = load_data(PATH_VANILLA)
glasses = load_data(PATH_GLASSES)
masks = load_data(PATH_MASKS)

uh_vanilla = load_data(UH_PATH_VANILLA)
uh_glasses = load_data(UH_PATH_GLASSES)
uh_masks = load_data(UH_PATH_MASKS)

ub_vanilla = load_data(UB_PATH_VANILLA)
ub_glasses = load_data(UB_PATH_GLASSES)
ub_masks = load_data(UB_PATH_MASKS)

l1_dict_glasses = {'l1_glasses_ours': [], 'l1_glasses_unih':[], 'l1_glasses_unib':[]}
l1_dict_masks = {'l1_masks_ours': [], 'l1_masks_unih':[], 'l1_masks_unib':[]}

def get_L1(vanilla, glasses, masks, key1, key2):
    for idx in range(len(vanilla)):
        l1_glasses = vanilla[idx] - glasses[idx]
        l1_masks = vanilla[idx] - masks[idx]

        l1_dict_glasses[key1].append(l1_glasses)
        l1_dict_masks[key2].append(l1_masks)


get_L1(vanilla, glasses, masks, 'l1_glasses_ours', 'l1_masks_ours')
get_L1(uh_vanilla, uh_glasses, uh_masks, 'l1_glasses_unih', 'l1_masks_unih')
get_L1(ub_vanilla, ub_glasses, ub_masks, 'l1_glasses_unib', 'l1_masks_unib')

l1_plot = Plotter(
    datasets=[l1_dict_masks['l1_masks_ours'], l1_dict_masks['l1_masks_unih'], l1_dict_masks[ 'l1_masks_unib']],
    labels=['251128e35', 'Unigaze-H', 'Unigaze-B'],
        )

l1_plot.plot_comparison(title='L1 on ZeroGaze (Masks)', range_val=(-15,15), save_path='L1-masks')

# PATH_ORIGINAL = "/data/sfhq_fluxdev_gazezero_v2/eval/unigaze_B_gaze_zerogaze_masks.npy"
# PATH_MIRRORED = "/data/sfhq_fluxdev_gazezero_v2/eval/unigaze_gaze_zerogaze_masks_mirrored.npy"

# orig = load_data(PATH_ORIGINAL)
# mirrored = load_data(PATH_MIRRORED)

# l1_plot = Plotter(
#     datasets=[orig, mirrored],
#     labels=['Original', 'Mirrored'],
#         )

# l1_plot.plot_comparison(title='Zerogaze(Vanilla)', range_val=(-20,20), save_path='unigaze_b_masks')