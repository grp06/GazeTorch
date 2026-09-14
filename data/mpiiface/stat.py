'''
plot histogram of yaw & pitch
'''

import numpy as np
import os
import matplotlib.pyplot as plt

gt_path = "/data/mpiiface/gt_mpiiface_with_conf_v2.npy"
D = np.load(gt_path, allow_pickle=True).item()
gt = D['gaze']

p = gt[:,0]
y = gt[:,1]

def hist(v, target_path, label):
    frq, edges = np.histogram(v, bins=16, range=(-np.pi*32/180, np.pi*32/180)) # 16 bins, -32~32, 4 deg per bin
    edges *= 180/np.pi
    plt.bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge");
    plt.title(label)
    plt.savefig(target_path)

hist(p, "temp_results/mpii/hist_p.png", "pitch")
hist(y, "temp_results/mpii/hist_y.png", "yaw")
print("Voila.")