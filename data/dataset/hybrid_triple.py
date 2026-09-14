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
from data.dataset.gazegene import Dataset_GazeGene
from data.dataset.gaze360 import Dataset_Gaze360
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


class Dataset_HybridXGN():
    def __init__(self, path_gt_x=None, path_gt_g=None, path_gt_n=None,
                 split='train', augment=False, N=512,
        inputs = ['face'], labels = ['gaze'],
        augmenter_dir=consts.AUG_DIR_MATERIALS, augmenter_cfg_path=None,
        bin_args = {},
        GC3D_population = 500,
        debug = False,
        root_xgaze = None, root_gc3d = None, root_n = None,
        dataset_base=None,
        flip_mode = 'off', # works for multi-view mode
        **kwargs):
        '''
        The triple-hybrid dataset: X=XGaze, G=GC3D, N=GazeGene.
        N: the dataset length. In runtime,
            * N//3 will be drawn from XGaze's `id_list`,
            * N//3 will be from GC3D's, 
            * the rest will be drawn from GazeGene's `id_list`
            * shuffle() refreshes the `id_list` of the 3 datasets
            * for train, N should be divisible by (3 * batch_size * nbins_pitch * nbins_yaw)
            * for val, N should be divisible by (3 * (nbins_pitch-2) * (nbins_yaw-2))
        
        Assumption: when using 'bin', 'gaze' must be available too!
            bin id is based on gaze.

        "Hybrid"-exclusive attirbutes
        * dataset_id: xgaze / gc3d, specifies which dataset the sample belongs to
        '''
        self.hybrid_hbn = True
        args = arg_loader(_DEFAULT_ARGS, **kwargs)
        if split!='train':
            split_x = split_g = split_n = 'val'
        else:
            split_x = split_g = split_n = 'train'
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

        args_X = args.copy()
        if root_xgaze is not None:
            args_X['root'] = root_xgaze

        args_G = args.copy()
        if root_gc3d is not None:
            args_G['root'] = root_gc3d

        args_N = args.copy()
        if root_n is not None:
            args_N['root'] = root_n

        if dataset_base:
            dataset_base_X = dataset_base.DX
            dataset_base_G = dataset_base.DG
            dataset_base_N = dataset_base.DN
        else:
            dataset_base_X = None
            dataset_base_G = None
            dataset_base_N = None

        self.aug = initialize_augmenter(split, augmenter_dir, augmenter_cfg_path)
        print("[Dataset_HybridXGN]Augmenter loaded! Config = ", self.aug.path_cfg)
        
        print("[Dataset_HybridXGN]Loading XGaze dataset...")
        self.DX = Dataset_XGaze(path_gt=path_gt_x, split=split_x, augment=augment,
            inputs=inputs, labels=labels, bin_args=bin_args,
            target_N=N//3, dataset_base=dataset_base_X,
            child_of_hybrid=True,
            augmenter_base=self.aug,
             **args_X)
        print("[Dataset_HybridXGN]XGaze loaded! Len =", len(self.DX))
        if debug:
            # GC3D loading is too slow, skip it
            from copy import deepcopy
            self.DG = deepcopy(self.DX)
            self.DN = deepcopy(self.DG)
            print("[Dataset_HybridXGN]Debug mode. self.DN == self.DG == self.DX")
        else:
            print("[Dataset_HybridXGN]Loading GC3D dataset...")
            self.DG = Dataset_GC3d(path_gt=path_gt_g, split=split_g, augment=augment,
                inputs=inputs, labels=labels, bin_args=bin_args,
                population = GC3D_population,
                target_N=N//3, dataset_base=dataset_base_G,
                child_of_hybrid=True,
                augmenter_base=self.aug,
                **args_G)
            print("[Dataset_HybridXGN]GC3D loaded! Len =", len(self.DG))

            print("[Dataset_HybridXGN]Loading GazeGene dataset...")
            self.DN = Dataset_GazeGene(path_gt=path_gt_n, split=split_n, augment=augment,
                inputs=inputs, labels=labels, bin_args=bin_args,
                target_N=N//3, dataset_base=dataset_base_N,
                child_of_hybrid=True,
                augmenter_base=self.aug,
                **args_N)
            print("[Dataset_HybridXGN]GazeGene loaded! Len =", len(self.DN))

        # self.aug = initialize_augmenter(split, augmenter_dir, augmenter_cfg_path)
        # self.DX.aug = self.aug
        # self.DG.aug = self.aug
        # self.DN.aug = self.aug

        self.id_list = np.concatenate((self.DX.id_list, self.DG.id_list, self.DN.id_list), axis=0)

        self.debug = debug
        self.nbins_pitch = bin_args['pitch_nbins']
        self.nbins_yaw = bin_args['yaw_nbins']
        self.pitch_bias = bin_args['pitch_bias']
        self.bin_size = bin_args.get('bin_size', 4)

        print("[hybrid]bin_args:", bin_args)

        self.limit_pitch_u = (self.pitch_bias + self.nbins_pitch / 2 * self.bin_size) * np.pi / 180
        self.limit_pitch_l = (self.pitch_bias - self.nbins_pitch / 2 * self.bin_size) * np.pi / 180
        self.limit_yaw_u = (self.nbins_yaw / 2) * self.bin_size * np.pi / 180
        self.limit_yaw_l = (-self.nbins_yaw / 2) * self.bin_size * np.pi / 180

        # n_DX and n_DG should be roughly equal
        # self.n_DX = len(self.DX)
        # self.n_DG = len(self.DG)
        self.N = 768 if self.debug else N

        self.set_flip_mode(flip_mode)

    def __len__(self):
        return self.N

    def __getitem__(self, i, return_dict=False): # set default value
        if i<self.N // 3:
            id_eq = i
            obj = self.DX
            dataset_id = 0 # XGaze
        elif self.N // 3 <= i < (self.N // 3)*2:
            id_eq = i-self.N//3
            obj = self.DG
            dataset_id = 1 # GC3D
        else:
            id_eq = i-(self.N//3)*2
            obj = self.DN
            dataset_id = 2 # GazeGene
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
        self.DN.shuffle()
        self.id_list = np.concatenate((self.DX.id_list, self.DG.id_list, self.DN.id_list), axis=0)

    def set_flip_mode(self, flip_mode): # works for multi-view mode
        self.DX.flip_mode = flip_mode
        self.DG.flip_mode = flip_mode
        self.DN.flip_mode = flip_mode

    def update_aug_states(self):
        self.DX.aug = self.aug
        self.DG.aug = self.aug
        self.DN.aug = self.aug

        self.DX.update_aug_states()
        self.DG.update_aug_states()
        self.DN.update_aug_states()


class Dataset_XG360():
    def __init__(self, path_gt_x=None, path_gt_g=None, path_gt_360=None, split='train', augment=False, N=512,
        inputs = ['face'], labels = ['gaze'],
        augmenter_dir=consts.AUG_DIR_MATERIALS, augmenter_cfg_path=None,
        bin_args = {},
        # bin_size = 4,
        # pitch_bias = -12, pitch_nbins = 15,
        # yaw_bias = 0, yaw_nbins = 17,
        GC3D_population = 500,
        debug = False,
        root_xgaze = None, root_gc3d = None, root_360 = None,
        dataset_base=None,
        flip_mode = 'off', # works for multi-view mode
        **kwargs):
        '''
        N be the dataset length. In runtime,
            * N//3 will be drawn from XGaze's `id_list` and
            * N//3 will be drawn from GC3D's `id_list
            * the rest N//3 will be drawn from Gaze360's `id_list`
            * shuffle() refreshes the `id_list` of both datasets
        
        Assumption: when using 'bin', 'gaze' must be available too!
            bin id is based on gaze.

        "Hybrid"-exclusive attirbutes
        * dataset_id: xgaze / gc3d, specifies which dataset the sample belongs to
        '''
        self.hybrid360 = True
        args = arg_loader(_DEFAULT_ARGS, **kwargs)
        if split!='train':
            split_x = split_g = split_360 = 'val'
        else:
            split_x = split_g = split_360 = 'train'
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

        args_X = args.copy()
        if root_xgaze is not None:
            args_X['root'] = root_xgaze

        args_G = args.copy()
        if root_gc3d is not None:
            args_G['root'] = root_gc3d

        args_360 = args.copy()
        if root_360 is not None:
            args_360['root'] = root_360

        if dataset_base:
            dataset_base_X = dataset_base.DX
            dataset_base_G = dataset_base.DG
            dataset_base_360 = dataset_base.D360
        else:
            dataset_base_X = None
            dataset_base_G = None
            dataset_base_360 = None
        
        print("[Dataset_Hybrid_Triple]Loading XGaze dataset...")
        self.DX = Dataset_XGaze(path_gt=path_gt_x, split=split_x, augment=augment,
            inputs=inputs, labels=labels, bin_args=bin_args,
            target_N=N//3, dataset_base=dataset_base_X,
             **args_X)
        print("[Dataset_Hybrid_Triple]XGaze loaded! Len =", len(self.DX))
        if debug:
            # GC3D loading is too slow, skip it
            from copy import deepcopy
            self.DG = deepcopy(self.DX)
            print("[Dataset_Hybrid_Triple]Debug mode. self.DG == self.DX")
        else:
            print("[Dataset_Hybrid_Triple]Loading GC3D dataset...")
            self.DG = Dataset_GC3d(path_gt=path_gt_g, split=split_g, augment=augment,
                inputs=inputs, labels=labels, bin_args=bin_args,
                population = GC3D_population,
                target_N=N//3, dataset_base=dataset_base_G,
                child_of_hybrid=True,
                **args_G)
            print("[Dataset_Hybrid_Triple]GC3D loaded! Len =", len(self.DG))

        self.D360 = Dataset_Gaze360(split=split_360, augment=augment,
            inputs=inputs, labels=labels, bin_args=bin_args,
            target_N=N//3, dataset_base=dataset_base_360,
            child_of_hybrid=True,
             **args_360)
        print("[Dataset_Hybrid_Triple]Gaze360 loaded! Len =", len(self.D360))

        self.aug = initialize_augmenter(split, augmenter_dir, augmenter_cfg_path)
        self.DX.aug = self.aug
        self.DG.aug = self.aug
        self.D360.aug = self.aug
        self.id_list = np.concatenate((self.DX.id_list, self.DG.id_list, self.D360.id_list), axis=0)

        self.debug = debug
        self.nbins_pitch = bin_args['pitch_nbins']
        self.nbins_yaw = bin_args['yaw_nbins']
        self.pitch_bias = bin_args['pitch_bias']
        self.bin_size = bin_args.get('bin_size', 4)

        self.limit_pitch_u = (self.pitch_bias + self.nbins_pitch / 2 * self.bin_size) * np.pi / 180
        self.limit_pitch_l = (self.pitch_bias - self.nbins_pitch / 2 * self.bin_size) * np.pi / 180
        self.limit_yaw_u = (self.nbins_yaw / 2) * self.bin_size * np.pi / 180
        self.limit_yaw_l = (-self.nbins_yaw / 2) * self.bin_size * np.pi / 180

        # n_DX and n_DG should be roughly equal
        # self.n_DX = len(self.DX)
        # self.n_DG = len(self.DG)
        self.N = 768 if self.debug else N

        self.set_flip_mode(flip_mode)

    def __len__(self):
        return self.N

    def __getitem__(self, i, return_dict=False): # set default value
        if i<self.N // 3:
            id_eq = i
            obj = self.DX
            dataset_id = 0 # XGaze
        elif self.N // 3 < i < (self.N // 3)*2:
            id_eq = i-self.N//3
            obj = self.DG
            dataset_id = 1 # GC3D
        else:
            id_eq = i-(self.N//3)*2
            obj = self.D360
            dataset_id = 0 # Gaze360
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
        self.D360.shuffle()
        self.id_list = np.concatenate((self.DX.id_list, self.DG.id_list, self.D360.id_list), axis=0)

    def set_flip_mode(self, flip_mode): # works for multi-view mode
        self.DX.flip_mode = flip_mode
        self.DG.flip_mode = flip_mode
        self.D360.flip_mode = flip_mode
