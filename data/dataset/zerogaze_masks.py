import os
import numpy as np
import random
import cv2

from data.augmenter.augmenter import initialize_augmenter
import data.consts as consts

from torch.utils.data import Dataset as D

ROOT = '/data/sfhq_fluxdev_gazezero_v2/'
PATH_GT =  "/data/sfhq_fluxdev_gazezero_v2/valid_ids.npy"

_DIRS = {
    'face': "gazezero_masks_224", 
    'glasses': None,
    'facemask': None,
    'greenmask': None,
}

PATH_SUBJECT_LMK = None

class Dataset_ZeroGaze_Masks(D):
    def __init__(self, root=ROOT, path_gt=None, split='train', augment=False, debug=False, 
            inputs = ['face'], labels = ['gaze'], mirrored = False,
            rgb_no_normalization = False, return_tensor=False,
            skip_val=False, 
            img_channel_chw=True,
            augmenter_dir=consts.AUG_DIR_MATERIALS, augmenter_cfg_path=None,
            augmenter_options:dict={},
            **kwargs):
        '''
        root: dataset root dir
        path_npy: ground truth location
        split: train/val/all/N
        dir_* (**kwargs in the code): images location (can be face/eye/eyemap/irismap/facemap/facemask)
        debug: if True, the dataset only contains 512 samples
        img_channel_chw: if True, all images are in C*H*W format (PyTorch default)
        return_tensor: if True, apply torch.from_numpy() so that output values are Tensors.
    
        total valid_ids: 83818
        '''
        self.split = split
        if split in ['train', 'val']:
            self.start_val = 67055 # train over val - 8:2
            self.num = 83818
            self.dirs = _DIRS
            self.skip_val = skip_val
            if path_gt is None:
                path_gt = PATH_GT
            if self.skip_val:
                print("[ZeroGaze]Warning: skip_val = True. Will use val data for training!")
                self.id_list_o = list(range(self.num))
            else:
                self.id_list_o = list(range(self.start_val)) if self.split=='train' else list(range(self.start_val, self.num))
        elif split in ['all']:
            self.num = 83818
            self.id_list_o = list(range(0, self.num))
            self.dirs = _DIRS
            path_gt = PATH_GT
        else:
            self.num = split
            self.id_list_o = list(range(0, self.num))
            self.dirs = _DIRS
            path_gt = PATH_GT


        path_npy = os.path.join(root, path_gt)
        # This only contains the valid_ids of the dataset
        self.valid_ids = np.load(path_npy, allow_pickle=True)

        # The gaze labels for the gt should be 0,0
        self.gt_data = {}
        self.gt_data['gaze'] = np.zeros((self.num, 2))

        self.inputs = inputs
        self.labels = labels
        self.all = inputs + labels
        for k, v in kwargs.items():
            if k.startswith("dir_"):
                name = k[4:]
                self.dirs[name] = v

        self.root = root
        self.debug = debug

        self.id_list = self._filter_ids()

        self.chw = img_channel_chw
        self.rgb_no_norm = rgb_no_normalization
        self.return_tensor = return_tensor

        self.num_data = len(self.id_list)

        self.augment = augment and self.split=='train'
        augment_split = 'train' if self.augment else 'test'
        self.augment_args = augmenter_options
        self.aug = initialize_augmenter(augment_split, augmenter_dir, augmenter_cfg_path)

        self.augment_glasses = False
        self.face_apply_mask = False
        self.face_replace_green = False
        self.mirrored = mirrored
        
    def __len__(self):
        if self.debug:
            return 512
        return self.num_data

    def _filter_ids(self):
        id_list_new = []
        for i in self.id_list_o:
            valid_flag = True
            for name in consts.IMG_TYPES:
                if name not in self.all: continue
                ext = '.png'
                path = os.path.join(self.root, self.dirs[name], str(i)+ext)
                if not os.path.isfile(path):
                    valid_flag = False
                    break
            if valid_flag:
                id_list_new.append(i)
        self.num_data = len(id_list_new)
        print("[Dataset]Valid ID list generated. # =", self.num_data)
        return id_list_new

    def _load_rgb(self, type:str, i:int, mixup_id=None, alpha=0.0, **kwargs):
        ext = ".png" # usually face is in .jpg format but for sfhq, it's in .png
        path = os.path.join(self.root, self.dirs[type], str(i)+ext)
        if not os.path.isfile(path):
            print("[Dataset]Warning: image {} not exists. Return None.".format(path))
            return None
        rgb = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if type in consts.RGB_TYPES:
            rgb = rgb[:,:,::-1] # Reverse because openCV stores in BGR instead of RGB

        if not type.startswith("face") and self.eye_resize_320:
            rgb = cv2.resize(rgb, (320, 192)) 

        if type.startswith('irismap') or type.startswith('eyemap'):
            if rgb.ndim > 2:
                rgb = rgb[..., 0]
            rgb = rgb[..., None] # HxWx1
            return rgb
        
        if self.mirrored:
            rgb = cv2.flip(rgb, 1)

        return rgb

    def __getitem__(self, i, return_dict=False): 
        index = self.id_list[i]
        id_eq = index
        
        values = {}
        # load each value involved
        img_dicts = {}
        for cat in consts.CATEGORIES:
            img_dicts[cat] = {}

        for name in self.all: # self.all <- input + labels 
            if name in consts.ANGULAR_TYPES:
                val = self.gt_data[name][index]
                val = val.astype(np.float32)
                values[name] = val
            elif name == 'pitch':
                val = self.gt_data['gaze'][index][0]
                val = val.astype(np.float32)
                values[name] = val
            elif name == 'id':
                values_id = index
                values[name] = values_id
            elif name in consts.CONST_TYPES or name.startswith('w'): # fields that don't change when flipping
                val = self.gt_data[name][index]
                values[name] = val
            else: #image
                cat = consts.get_category(name)
                img_dicts[cat][name] = self._load_rgb(name, id_eq)
        
        img_mask = None
        green_mask = None

        # batch-augment
        if not self.rgb_no_norm:
            if self.augment:
                for cat in consts.CATEGORIES:
                    if len(img_dicts[cat]) == 0: continue
                    subject_landmarks, subject_headpose = None, None
                    
                    img_dicts[cat] = self.aug.augment(None, 
                        add_mask=self.face_apply_mask, rgb_mask=img_mask,
                        replace_green=self.face_replace_green, rgb_green=green_mask,
                        img_dict=img_dicts[cat],
                        subject_landmarks=subject_landmarks, subject_headpose=subject_headpose, 
                        **self.augment_args)
            else:
                for cat in consts.CATEGORIES:
                    if len(img_dicts[cat]) == 0: continue
                    img_dicts[cat] = self.aug.augment(None,
                        img_dict=img_dicts[cat],
                        **self.augment_args)
        
        # transpose & unpack img_dict => values
        for cat in consts.CATEGORIES:
            if self.chw: # HWC (TF, numpy) => CHW (PyTorch)
                for key, rgb in img_dicts[cat].items():
                    if key == 'aug': continue # override the 'aug' switch in augmenter.py
                    if rgb.ndim == 3:
                        img_dicts[cat][key] = rgb.transpose((2,0,1))
            values.update(img_dicts[cat])

        # turn np arrays into torch tensors
        if self.return_tensor:
            import torch
            for key, value in values.items():
                if isinstance(value, np.ndarray):
                    values[key] = torch.from_numpy(value.copy())

        if return_dict:
            return values

        # load into I/O tensors
        Xs = [values[name] for name in self.inputs]
        labels = [values[name] for name in self.labels]

        return Xs, labels

    def shuffle(self):
        random.shuffle(self.id_list)
        print("[Dataset]Shuffled.")
