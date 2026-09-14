'''
Eyediap dataset class.
Apr '23: updated by Julia
    * reworked 3D gaze labels
    * new arg `session`: "CS/DS" support
'''
import os
import cv2
import torch
import random
import numpy as np

from data.augmenter.augmenter import initialize_augmenter
import data.consts as consts

from torch.utils.data import Dataset as D

ROOT = '/data/eyediap/'

# if using Beihang dataset use or pass from .yaml:
# PATH_GT = ['gt_eyediap_cs_with_conf_v3_bh.npy', 'gt_eyediap_ds_with_conf_v3_bh.npy']
# _DIRS = {
#     'face': ['normalized_face_224_cs_bh', 'normalized_face_224_ds_bh']
# }

PATH_GT = ['gt_eyediap_cs_with_conf_v3.npy', 'gt_eyediap_ds_with_conf_v3.npy']
FOLDS = [0,1,2,3]
_DIRS = {
    'face': ['normalized_face_224_cs', 'normalized_face_224_ds']
}
class Dataset_Eyediap(D):
    def __init__(self, root=ROOT, path_gt=PATH_GT, dirs=_DIRS,
            split='train', debug=False, augment=True,
            inputs = ['face'], labels = ['gaze'], session = ['cs','ds'],
            val_folds:list = [],
            rgb_no_normalization = False, return_tensor = False,
            flip_mode = 'off',
            img_channel_chw=True, 
            augmenter_dir=consts.AUG_DIR_MATERIALS, augmenter_cfg_path=None,
            augmenter_options:dict={},
            first_only = None,
            unigaze = '',
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
        return_tensor: if True, apply torch.from_numpy() so that output values are Tensors.
        augmenter*: see data.augmenter.augmenter.initialize_augmenter()

        Supported inputs & labels fields:
        face, gaze, headpose
        TODO: facemap, wE_L/R, wI_L/R

        Eyediap-exclusive attributes
        val_folds: the fold(s) (#0-3) for validation.
            so, if split==train, the Dataset uses the complement folds.
            If ==[], always apply all folds.
        session: continuous screen ('cs') ['cs'] or discrete screen ('ds') ['ds'] sessions or both ['cs', 'ds']
        '''
        # filter ground truth paths based on session
        gt = []
        if unigaze == 'cs':
            print("[EyeDiap]UniGaze CS mode...")
            gt = [np.load("/data/eyediap/gt_unigaze.npy", allow_pickle=True).item()]
        elif unigaze == 'ds':
            print("[EyeDiap]UniGaze DS mode...")
            gt = [np.load("/data/eyediap/gt_unigaze_ds.npy", allow_pickle=True).item()]
        else:
            for path in path_gt:
                if any(s in path for s in session):
                    path_npy = os.path.join(root, path)
                    gt.append(np.load(path_npy, allow_pickle=True).item())
        
        # 2 sessions ['cs', 'ds']
        # merge ground truth data
        if len(gt) == 2:
            self.gt_data = {key: gt[0][key] + gt[1][key] for key in gt[0]}
        # 1 session ['cs'] or ['ds']
        else:
            self.gt_data = gt[0]

        self.inputs = inputs
        self.labels = labels
        self.all = inputs+labels

        self.dirs = dirs
        for k, v in kwargs.items():
            if k.startswith("dir_"):
                name = k[4:]
                self.dirs[name] = v
        
        # filter input (face) directory based on session
        for dir_type, dir_names in self.dirs.items():
            for dir_name in dir_names:
                if not any(s in dir_name for s in session):
                    self.dirs[dir_type].remove(dir_name)
        
        self.root = ROOT
        self.split = split
        self.debug = debug
        self.num_data_o = len(self.gt_data['fn'])
        self.id_list_o = list(range(self.num_data_o))

        self.flip_mode = flip_mode if self.split=='train' else 'off'
        self.folds = val_folds
        self.chw = img_channel_chw
        self.rgb_no_norm = rgb_no_normalization
        self.return_tensor = return_tensor
        
        if unigaze!='':
            self.num_data = self.num_data_o
        else:
            self.valid_ids = self._filter_ids()

        if first_only is not None:
            self.num_data = min(self.num_data, first_only)
        self.id_list = list(range(self.num_data))

        self.augment = augment and self.split=='train'
        self.augment_args = augmenter_options
        self.aug = initialize_augmenter(self.split, augmenter_dir, augmenter_cfg_path)

    def __len__(self):
        if self.debug:
            return 512
        return self.num_data

    def _filter_ids(self):
        valid_ids = []
        if len(self.folds) > 0:
            if self.split=='train':
                valid_folds = [f for f in FOLDS if f not in self.folds]
            else:
                valid_folds = self.folds
            for i in self.id_list_o:
                fold_id = self.gt_data['fold_id'][i]
                if fold_id not in valid_folds:
                    continue
                valid_ids.append(i)
        else:
            valid_ids = self.id_list_o
        self.num_data = len(valid_ids)
        print("[Dataset]Valid ID list generated. # =", self.num_data)
        return valid_ids

    def _load_rgb(self, type:str, i:int, **kwargs):
        '''
        type in ("face")
        We have face images only at the moment.
        '''
        for name in self.dirs[type]:
            fn = self.gt_data['fn'][i]
            path = os.path.join(self.root, name, f'{fn}.png')
            if not os.path.isfile(path):
                continue
            bgr = cv2.imread(path, 1)

            rgb = bgr[:, :, ::-1]

            return rgb

        print("[Dataset]Warning: image {} not exists. Return None.".format(path))
        return None

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
            if name in consts.ANGULAR_TYPES:
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
    d = Dataset_Eyediap()
    Xs, labels = d.__getitem__(2)
    