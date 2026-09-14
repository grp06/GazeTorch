'''
XGaze + GazeCapture hybrid dataset.

Pitch
XGaze: 17 pitch bins, bias=-16
GC3D: 17 pitch bins, bias=-8
=> Hybrid: 15 pitch bins, bias=-12
* so, for XGaze, Hybrid = pitch bin [2:] ;
       for GC3D, Hybrid = pitch bin [:-2] ;

Yaw
17 yaw bins, bias=0
'''
from typing import Iterable
from data.dataset.xgaze import Dataset_XGaze
from data.dataset.gc3d import Dataset_GC3d
import data.consts as consts
from tools.utils import arg_loader
from data.augmenter.augmenter import initialize_augmenter

import os
import numpy as np
import random
import cv2

def _args_to_dict(**kwargs):
    return kwargs

def where(L:Iterable, obj):
    for i in range(len(L)):
        if L[i]==obj:
            return i
    return -1

_DEFAULT_ARGS = {
    "bin_sampler": True,
    "bin_noclip": True,
    "bin_limit": False,
}
class Dataset_XplusG():
    def __init__(self, path_gt_x=None, path_gt_g=None, split='train', augment=False, N=512,
        inputs = ['face'], labels = ['gaze'],
        augmenter_dir=consts.AUG_DIR_MATERIALS, augmenter_cfg_path=None,
        bin_args = {},
        # bin_size = 4,
        # pitch_bias = -12, pitch_nbins = 15,
        # yaw_bias = 0, yaw_nbins = 17,
        GC3D_population = 500,
        debug = False,
        root_xgaze = None, root_gc3d = None,
        dataset_base=None,
        flip_mode = 'off', # works for multi-view mode
        **kwargs):
        '''
        N be the dataset length. In runtime,
            * N/2 will be drawn from XGaze's `id_list` and
            * the rest N/2 will be drawn from GC3D's `id_list`
            * shuffle() refreshes the `id_list` of both datasets
            * for train, N should be divisible by (3 * batch_size * nbins_pitch * nbins_yaw)
            * for val, N should be divisible by (3 * batch_size * (nbins_pitch-2) * (nbins_yaw-2))
        
        Assumption: when using 'bin', 'gaze' must be available too!
            bin id is based on gaze.

        "Hybrid"-exclusive attirbutes
        * dataset_id: xgaze / gc3d, specifies which dataset the sample belongs to
        '''
        self.hybrid = True
        args = arg_loader(_DEFAULT_ARGS, **kwargs)
        if split!='train':
            split_x = 'val'
            split_g = 'val'
        else:
            split_x = split_g = 'train'
        self.split = split

        self.inputs = inputs.copy()
        self.labels = labels.copy()
        self.all = self.inputs + self.labels

        if 'dataset_id' in self.all:
            self.use_dataset_id = True
            for L in (inputs, labels):
                if 'dataset_id' in L:
                    L.remove('dataset_id')
        else:
            self.use_dataset_id = False

        args['debug'] = False
        args_X = args.copy()
        if root_xgaze is not None:
            args_X['root'] = root_xgaze

        args_G = args.copy()
        if root_gc3d is not None:
            args_G['root'] = root_gc3d

        if dataset_base:
            dataset_base_X = dataset_base.DX
            dataset_base_G = dataset_base.DG
        else:
            dataset_base_X = None
            dataset_base_G = None
        
        print("[Dataset_Hybrid]Loading XGaze dataset...")
        self.DX = Dataset_XGaze(path_gt=path_gt_x, split=split_x, augment=augment,
            inputs=inputs, labels=labels, bin_args=bin_args,
            target_N=N//2, dataset_base=dataset_base_X,
             **args_X)
        print("[Dataset_Hybrid]XGaze loaded! Len =", len(self.DX))
        if debug:
            # GC3D loading is too slow, skip it
            from copy import deepcopy
            self.DG = deepcopy(self.DX)
            print("[Dataset_Hybrid]Debug mode. self.DG == self.DX")
        else:
            print("[Dataset_Hybrid]Loading GC3D dataset...")
            self.DG = Dataset_GC3d(path_gt=path_gt_g, split=split_g, augment=augment,
                inputs=inputs, labels=labels, bin_args=bin_args,
                population = GC3D_population,
                target_N=N//2, dataset_base=dataset_base_G,
                child_of_hybrid=True,
                **args_G)
            print("[Dataset_Hybrid]GC3D loaded! Len =", len(self.DG))

        self.aug = initialize_augmenter(split, augmenter_dir, augmenter_cfg_path)
        self.DX.aug = self.aug
        self.DG.aug = self.aug
        self.id_list = np.concatenate((self.DX.id_list, self.DG.id_list), axis=0)

        self.debug = debug
        self.nbins_pitch = bin_args['pitch_nbins']
        self.nbins_yaw = bin_args['yaw_nbins']
        self.pitch_bias = bin_args['pitch_bias']
        self.bin_size = bin_args.get('bin_size', 4)

        self.limit_pitch_u = (self.pitch_bias + self.nbins_pitch / 2 * self.bin_size) * np.pi / 180
        self.limit_pitch_l = (self.pitch_bias - self.nbins_pitch / 2 * self.bin_size) * np.pi / 180
        self.limit_yaw_u = (self.nbins_yaw / 2) * self.bin_size * np.pi / 180
        self.limit_yaw_l = (-self.nbins_yaw / 2) * self.bin_size * np.pi / 180

        self.N = 512 if self.debug else N

        self.set_flip_mode(flip_mode)

    def __len__(self):
        return self.N

    def __getitem__(self, i, return_dict=False): # set default value
        if i<self.N // 2:
            id_eq = i
            id_eq = id_eq % len(self.DX) # in case the dataset is not long enough
            obj = self.DX
            dataset_id = 0 # XGaze
        else:
            id_eq = i-self.N//2
            id_eq = id_eq % len(self.DG)
            obj = self.DG
            dataset_id = 1 # GC3D
        values = obj.__getitem__(id_eq, True)

        # re-clip gaze value
        if 'bin' in self.all:
            # discard original 'bin' value but compute again here
            g = values['gaze']
            pitch_cid = int(np.clip((g[0] * 180. / np.pi - self.limit_pitch_l * 180. / np.pi) // self.bin_size, 0, self.nbins_pitch-1)) # positive/negative changed before self.limit_...
            yaw_cid = int(np.clip((g[1] * 180. / np.pi - self.limit_yaw_l * 180. / np.pi) // self.bin_size, 0, self.nbins_yaw-1))
            values['bin'] = np.array([pitch_cid, yaw_cid])
            
        # if 'gaze' in self.all:
        #     g = values['gaze']
        #     g[0] = np.clip(g[0], self.limit_pitch_l, self.limit_pitch_u)
        #     g[1] = np.clip(g[1], self.limit_yaw_l, self.limit_yaw_u)
        #     values['gaze'] = g

        if self.use_dataset_id:
            values['dataset_id'] = dataset_id

        if return_dict:
            return values

        # load into I/O tensors
        Xs = [values[name] for name in self.inputs]
        labels = [values[name] for name in self.labels]

        return Xs, labels

    def shuffle(self):
        self.DX.shuffle()
        self.DG.shuffle()
        self.id_list = np.concatenate((self.DX.id_list, self.DG.id_list), axis=0)

    def set_flip_mode(self, flip_mode): # works for multi-view mode
        self.DX.flip_mode = flip_mode
        self.DG.flip_mode = flip_mode