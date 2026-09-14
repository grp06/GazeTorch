'''
RT_Gene dataset class.
'''
import os
import cv2
import torch
import random
import numpy as np

from data.augmenter.augmenter import initialize_augmenter
import data.consts as consts

from torch.utils.data import Dataset as D

ROOT = '/data/rt_gene/'
PATH_GT = 'gt_rt_gene_with_conf_v2.npy'
_DIRS = {
    'face': 'face_224',
}
class Dataset_RT_GENE(D):
    def __init__(self, root=ROOT, path_gt=PATH_GT,
            split='train', debug=False, augment=True,
            inputs = ['face'], labels = ['gaze'], 
            rgb_no_normalization = False, return_tensor = False,
            flip_mode = 'off',
            img_channel_chw=True, 
            augmenter_dir=consts.AUG_DIR_MATERIALS, augmenter_cfg_path=None,
            augmenter_options:dict={},
            enable_w_wrt_subject_id=False,
            **kwargs):
        '''
        split: train/val
        debug: if True, the dataset has much less samples so as to run trainning epochs faster
        augment: if True and split=='train', switch on data augmentation
        dir_*: images location (can be face/eye/eyemap/irismap/facemap/facemask)
        flip_mode: can be "off"(default)/"on"/"random". for training only. 
            * "off": never apply horizontal flip.
            * "on": always flip.
            * "random": 50% probability to flip.
        img_channel_chw: if True, all images are in C*H*W format (PyTorch default)
        augmenter_options: a dict of customized arguments to be passed to the Augmenter during __getitem__().
            * In default, consts.AUG_DEFAULT_ARGS_ON(augment=True)/OFF(False) will be passed
            * If specified, the default arguments will be overwritten accordingly
            * Add "augmenter_options" node in dataset_train/val.arg sections in Config files
        return_tensor: if True, apply torch.from_numpy() so that output values are Tensors.

        Supported inputs & labels fields:
        face, gaze, headpose
        TODO: facemap, wE_L/R, wI_L/R
        '''
        
        path_npy = os.path.join(root, path_gt)
        self.gt_data = np.load(path_npy, allow_pickle=True).item()

        self.dirs = _DIRS
        self.inputs = inputs
        self.labels = labels
        self.enable_w_wrt_subject_id = enable_w_wrt_subject_id
        if self.enable_w_wrt_subject_id:
            self.labels += ['w_wrt_subject_id']
   
        self.all = inputs+labels
        
        self.root = ROOT
        self.split = split
        self.debug = debug
        self.num_data_o = len(self.gt_data['fn'])
        self.id_list_o = list(range(self.num_data_o))

        self.flip_mode = flip_mode if self.split=='train' else 'off'
        self.chw = img_channel_chw
        self.rgb_no_norm = rgb_no_normalization
        self.return_tensor = return_tensor
        
        self.valid_ids = self._filter_ids()
        self.id_list = list(range(self.num_data))

        self.augment = augment and self.split=='train'
        self.augment_args = augmenter_options
        self.aug = initialize_augmenter(self.split, augmenter_dir, augmenter_cfg_path)

    def __len__(self):
        if self.debug:
            return 512
        return self.num_data

    def _filter_ids(self):
        # can be written later (placeholder)
        valid_ids = self.id_list_o
        self.num_data = len(valid_ids)
        return valid_ids

    def _load_rgb(self, type:str, i:int, **kwargs):
        '''
        type in ("face")
        We have face images only at the moment.
        '''
        path = os.path.join(self.root, self.dirs[type], self.gt_data['fn'][i]+'.png')
        if not os.path.isfile(path):
            print("[Dataset]Warning: image {} not exists. Return None.".format(path))
            return None
        bgr = cv2.imread(path, 1)

        rgb = bgr[:, :, ::-1]

        return rgb

    def __getitem__(self, i):     
        index = self.id_list[i]
        values = {}

        # global random values
        flip = False
        if self.split=='train':
            if self.flip_mode=='on' or (self.flip_mode=='random' and random.random()<0.5):
                flip = True

        img_dict = {}
        for name in self.all:
            if (name in (consts.ANGULAR_TYPES)) or (name == 'w_wrt_subject_id'):
                val = self.gt_data[name][index]
                val = val.astype(np.float32)
                values[name] = val
            else: #'face'
                img_dict[name] = self._load_rgb(name, index)

        # batch-augment
        if not self.rgb_no_norm and len(img_dict)>0:
            img_dict = self.aug.augment(None,
                                            img_dict=img_dict,
                                            flip=flip,
                                            **self.augment_args)

        # transpose & unpack img_dict => values
        if self.chw:  # HWC (TF, numpy) => CHW (PyTorch)
            for key, rgb in img_dict.items():
                if getattr(rgb, 'ndim', 1) == 3:
                    img_dict[key] = rgb.transpose((2, 0, 1))
        values.update(img_dict)

        if flip:
            for name in self.all:
                if name in consts.ANGULAR_TYPES:
                    values[name][1] = -values[name][1] # revert yaw value

        # turn np arrays into torch tensors
        if self.return_tensor:
            import torch
            for key, value in values.items():
                if isinstance(value, np.ndarray):
                    values[key] = torch.from_numpy(value.copy())

        # load into I/O tensors
        Xs = [values[name] for name in self.inputs]
        labels = [values[name] for name in self.labels]

        return Xs, labels

    def shuffle(self):
        self.id_list = np.random.permutation(self.id_list)

if __name__ == "__main__":
    d = Dataset_RT_GENE()
    Xs, labels = d.__getitem__(2)