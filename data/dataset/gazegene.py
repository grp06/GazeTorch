'''
GazeGene dataset
'''
import os
import cv2
import torch
import random
import numpy as np

from data.augmenter.augmenter import initialize_augmenter, process_getitem_values
import data.consts as consts

from torch.utils.data import Dataset as D

from data.gazegene.gazegene_2d_sampler import GazeGene_2D_Sampler


ROOT = '/data/GazeGene/'
PATH_SUBJECT_LMK = '/data/GazeGene/lmk.npy'
PATH_CAM_NUM = '/data/GazeGene/gazegene_camera.npy'

PATH_GT = 'gazegene_gt_conf_v2.npy' # gt file dict_keys(['fn', 'subject_cid', 'cam_id', 'gaze', 'headpose'])
PATH_ID = 'valid_ids_roll_pi36.npy'

DEFAULT_AUG_ARGS_PATH = 'config/augmenter/default_train.yaml' # use default train augmenter yaml

_DIRS = {
    'face': 'face_224',
    'facemap_v3': 'seg',
    'facemask': 'facemask_224',
    'greenmask': 'backgrounds_mp',
}

class Dataset_GazeGene(D):
    def __init__(self, root=ROOT, path_gt=PATH_GT,
            split='train', debug=False, augment=True,
            inputs = ['face'], labels = ['gaze'], 
            rgb_no_normalization = False, return_tensor = False,
            flip_mode = 'off',
            img_channel_chw=True, 
            augmenter_base=None,
            augmenter_dir=consts.AUG_DIR_MATERIALS, augmenter_cfg_path=None,
            augmenter_options:dict={},
            enable_w_wrt_subject_id=False,
            dataset_base=None,
            bin_sampler=False,
            bin_args={},
            skip_val=False,
            target_N=None,
            child_of_hybrid=False,
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
        skip_val: (for training) if True, include all samples including val samples

        FACE attributes
        * face_apply_mask: if True, switch on "random facemask" in data aug (only when self.augment=True).
        * face_replace_green: if True, replace the green background with random solid color or an image (see data/gazegene/backgroundremover.py).
        * face_simulate_glasses: if True, simulate glasses reflection via Augmenter.Patcher

        Supported inputs & labels fields:
        face, gaze, headpose,
        facemap, wE_L/R, wI_L/R

        OTHER attributes
        * dataset_base: default=None. Otherwise, it should be an existing Dataset_XGaze instance. The new instance will load `gt_data` from it.
        * child_of_hybrid: should be True when it's part of a hybrid dataset instance. Otherwise False.
            * modifies the subject_cid to avoid overlapping with other datasets.
        '''
        self.root = ROOT
        self.split = split
        self.debug = debug
        self.dirs = _DIRS
        if path_gt is None:
            path_gt = PATH_GT

        # Augmenter
        self.augment = augment and self.split=='train'
        self.augment_args = augmenter_options
        if augmenter_cfg_path is None and self.split=='train':
            augmenter_cfg_path = DEFAULT_AUG_ARGS_PATH
        if augmenter_base is None:
            self.aug = initialize_augmenter(self.split, augmenter_dir, augmenter_cfg_path)
        else:
            self.aug = augmenter_base

        # [25.12.15] - Remove args for augmenter and only use augmenter config YAML file instead  
        self.update_aug_states()

        if dataset_base is None:
            path_npy = os.path.join(root, path_gt)
            self.gt_data = np.load(path_npy, allow_pickle=True).item()
            if self.augment_glasses and PATH_SUBJECT_LMK is not None:
                self.subject_landmarks = np.load(PATH_SUBJECT_LMK, allow_pickle=True)
                self.subject_headposes = self.gt_data['headpose']
                print("[GazeGene] Subject Landmarks (for glasses synth) Loaded.")
        else:
            # load from an existing dataset instance
            self.gt_data = dataset_base.gt_data
            if self.augment_glasses:
                self.subject_landmarks = dataset_base.subject_landmarks
                self.subject_headposes = dataset_base.subject_headposes
            print("[GazeGene]Loaded GT from an existing Dataset_GazeGene instance.")

        self.inputs = inputs
        self.labels = labels
        self.enable_w_wrt_subject_id = enable_w_wrt_subject_id
        if self.enable_w_wrt_subject_id:
            self.labels += ['w_wrt_subject_id']
   
        self.all = inputs+labels
        
        self.start_val_idx = 859429 # 51(51x18000) subjects for train, 5 subjects for val
        self.num_data_o = len(self.gt_data['fn'])
        valid_ids = np.load(os.path.join(self.root, PATH_ID), allow_pickle=True)
        self.skip_val = skip_val

        if self.skip_val:
            print("[GazeGene]Warning: skip_val = True. Will use val data for training!")
            self.id_list_o = [i for i in valid_ids]
        else:
            if split == 'train':
                self.id_list_o = [i for i in valid_ids if i < self.start_val_idx]
                # self.valid_ids[:self.start_val_idx]
            else:
                self.id_list_o = [i for i in valid_ids if i >= self.start_val_idx]
        self.id_list = self.id_list_o
        self.num_data = len(self.id_list)

        self.flip_mode = flip_mode if self.split=='train' else 'off'
        self.chw = img_channel_chw
        self.rgb_no_norm = rgb_no_normalization
        self.return_tensor = return_tensor
        self.child_of_hybrid = child_of_hybrid

        self.bin = bin_sampler

        self.nbins_pitch = bin_args['pitch_nbins']
        self.nbins_yaw = bin_args['yaw_nbins']
        self.pitch_bias = bin_args['pitch_bias']
        self.bin_size = bin_args.get('bin_size', 4)
        
        gaze = np.array(self.gt_data['gaze'])
        pitch_min = self.pitch_bias - self.nbins_pitch * self.bin_size / 2
        self.pitch_cid = np.clip((gaze[:, 0] * 180/np.pi - pitch_min) // self.bin_size, 0, self.nbins_pitch - 1).astype(int)

        yaw_min = 0 - self.nbins_yaw * self.bin_size / 2
        self.yaw_cid = np.clip((gaze[:, 1] * 180/np.pi - yaw_min) // self.bin_size, 0, self.nbins_yaw - 1).astype(int)
        self.limit_pitch_u = (self.pitch_bias + self.nbins_pitch / 2 * self.bin_size) * np.pi / 180
        self.limit_pitch_l = (self.pitch_bias - self.nbins_pitch / 2 * self.bin_size) * np.pi / 180
        self.limit_yaw_u = (self.nbins_yaw / 2) * self.bin_size * np.pi / 180
        self.limit_yaw_l = (-self.nbins_yaw / 2) * self.bin_size * np.pi / 180

        if self.bin:
            self.sampler = GazeGene_2D_Sampler(self.gt_data, self.id_list_o,
                                            split=split, 
                                            bin_args=bin_args,
                                            pitch_cid=self.pitch_cid, yaw_cid=self.yaw_cid,
                                            target_N=target_N)
            self.id_list = self.sampler.sample()
            self.num_data = len(self.id_list)
            self.target_N = target_N # should be identical or close to self.num_data

        print("[Dataset]Valid ID list generated. # =", self.num_data)

        # self.camera_check = np.load(PATH_CAM_NUM, allow_pickle=True) # seems not used?
 
    def __len__(self):
        if self.debug:
            return 512
        return len(self.id_list)

    def _load_rgb(self, type:str, i:int):
        '''
        type in ("face", "facemap", "facemap_v3")
        We have face images only at the moment.
        '''
        path = os.path.join(self.root, self.dirs[type], str(i) + '.png')
        if not os.path.isfile(path):
            print("[Dataset]Warning: image {} not exists. Return None.".format(path))
            return None
        rgb = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if type in consts.RGB_TYPES:
            rgb = rgb[:,:,::-1]

        return rgb

    def __getitem__(self, i, return_dict=False):     
        index = self.id_list[i]
        values = {}
        id_eq = self.gt_data['fn'][index]
        
        # global random values
        flip = False
        if self.split=='train':
            if self.flip_mode=='on' or (self.flip_mode=='random' and random.random()<0.5):
                flip = True

        img_dicts = {}
        for cat in consts.CATEGORIES:
            img_dicts[cat] = {}
        for name in self.all:
            if (name in (consts.ANGULAR_TYPES)) or (name == 'w_wrt_subject_id'):
                val = np.array(self.gt_data[name][index])
                val = val.astype(np.float32)
                values[name] = val
            elif name=="bin":
                pitch_bin_id = self.pitch_cid[index]
                yaw_bin_id = self.yaw_cid[index]
                if flip:
                    nbins_yaw = self.nbins_yaw
                    yaw_bin_id = -yaw_bin_id + nbins_yaw - 1
                values[name] = np.array([pitch_bin_id, yaw_bin_id])
            elif name == 'subject_cid':
                subject_cid = int(self.gt_data['subject_cid'][index])
                if self.child_of_hybrid:
                    subject_cid += 2000
                values[name] = subject_cid
            elif name in consts.CONST_TYPES or name.startswith('w'):
                if name == 'id':
                    val = index
                    values[name] = val
                else:
                    val = self.gt_data[name][index]
                    values[name] = val
            elif 'aug' in name:
                pass
            else: #'face'
                cat = consts.get_category(name)
                img_dicts[cat][name] = self._load_rgb(name, index)

        # Face mask
        img_mask = None
        if self.augment and self.face_apply_mask and 'face' in self.all:
            # load face mask
            img_mask_path = os.path.join(self.root, self.dirs['facemask'], str(index)+'.png')
            if os.path.isfile(img_mask_path):
                img_mask = cv2.imread(img_mask_path)

        # Background green mask
        green_mask = None
        if self.augment and self.face_replace_green and 'face' in self.all:
            # load background mask
            green_mask_path = os.path.join(self.root, self.dirs['greenmask'], str(index)+'.png')
            if os.path.isfile(green_mask_path):
                green_mask = cv2.imread(green_mask_path)

        # batch-augment
        if not self.rgb_no_norm:
            if self.augment:
                for cat in consts.CATEGORIES:
                    if len(img_dicts[cat]) == 0: continue
                    subject_landmarks, subject_headpose = None, None
                    if cat=='face':
                        if self.augment_glasses:
                            subject_landmarks = self.subject_landmarks[index]
                            subject_headpose = self.subject_headposes[index]

                    augmented = self.aug.augment(None,
                        add_mask=self.face_apply_mask, rgb_mask=img_mask, rgb_green=green_mask,
                        img_dict=img_dicts[cat],
                        flip=flip,
                        subject_landmarks=subject_landmarks, subject_headpose=subject_headpose,
                        **self.augment_args)
                    
                    img_dicts[cat] = augmented
                    if cat=='face':
                        process_getitem_values(values, self.all, augmented['aug'])
            else:
                for cat in consts.CATEGORIES:
                    if len(img_dicts[cat]) == 0: continue
                    img_dicts[cat] = self.aug.augment(None,
                                                    img_dict=img_dicts[cat],
                                                    flip=flip,
                                                    **self.augment_args)

        # transpose & unpack img_dict => values
        for cat in consts.CATEGORIES:
            if self.chw:  # HWC (TF, numpy) => CHW (PyTorch)
                for key, rgb in img_dicts[cat].items():
                    if getattr(rgb, 'ndim', 1) == 3:
                        img_dicts[cat][key] = rgb.transpose((2, 0, 1))
            values.update(img_dicts[cat])

        if flip:
            for name in self.all:
                if name in consts.ANGULAR_TYPES:
                    values[name][1] = -values[name][1] # revert yaw value

        # for "bin" case only
        if self.bin and "gaze" in values:
            g = values['gaze']
            g[0] = np.clip(g[0], self.limit_pitch_l, self.limit_pitch_u)
            g[1] = np.clip(g[1], self.limit_yaw_l, self.limit_yaw_u)
            values['gaze'] = g

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
        self.id_list = np.random.permutation(self.id_list)

    def update_aug_states(self):
        # for Joe's updates. Call this whenever an self.aug.* is changed externally.
        self.augment_glasses = self.aug.args.glasses.switch and self.split=='train' and PATH_SUBJECT_LMK is not None
        self.face_apply_mask = self.aug.args.add_mask.switch and self.dirs['facemask'] is not None and self.split=='train'
        self.face_replace_green = self.aug.args.replace_green.switch and self.dirs['greenmask'] is not None and self.split=='train'

if __name__ == "__main__":
    d = Dataset_GazeGene()
    Xs, labels = d.__getitem__(2)