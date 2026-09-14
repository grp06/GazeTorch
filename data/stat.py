'''
Usage: python dataset/stat.py -i <path> -s <sample_period=1>
Calculate mean & std of an image set (images stored in <path>/*.jpg)
'''
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="path")
parser.add_argument("-s", dest="sample", type=int, default=1)

def get_mean_std(fns, sample_period=1):
    s_mean = 0
    s_std = 0
    n = len(fns)
    c = 0
    for i in tqdm(range(0,n,sample_period)):
        fname = fns[i]
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float)/255.
        mean = np.mean(img, axis=(0,1))
        std = np.std(img, axis=(0,1))
        s_mean += mean
        s_std += std
        c += 1
    print('mean=', s_mean/c)
    print('std=', s_std/c)

def _gaze_hist(v, target_path, label):
    import matplotlib.pyplot as plt
    plt.figure()
    frq, edges = np.histogram(v, bins=32, range=(-np.pi*64/180, np.pi*64/180)) # 16 bins, -32~32, 4 deg per bin
    edges *= 180/np.pi
    plt.bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge");
    plt.title(label)
    plt.savefig(target_path)
    plt.close()

def gaze_histogram(path, target_dir):
    '''
    path: input NPY file storing a DICT containing gaze labels
    target_dir: output dir, will save "hist_p/y.png".
    '''
    gt = np.load(path, allow_pickle=True).item()
    gt = np.array(gt['gaze'])
    _gaze_hist(gt[:,0], os.path.join(target_dir, "hist_p.png"), 'pitch')
    _gaze_hist(gt[:,1], os.path.join(target_dir, "hist_y.png"), 'yaw')
    print("Voila.")
    
if __name__ == "__main__":
    # import glob
    # args = parser.parse_args()
    # fns = glob.glob('{}/*.jpg'.format(args.path))
    # get_mean_std(fns, args.sample)

    gaze_histogram("/data/lzh/alex2_full/gt_merged.npy", "temp_results/alex2")