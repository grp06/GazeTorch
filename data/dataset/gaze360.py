'''
Gaze360 dataset class. by Mehdi, initial version Oct 27 2022

Keys in ground truth NPY:
    * `fn`: filename
    * `split`: 'train', 'test' or 'validation'
    * `gaze_ecs`: Gaze in eye coordinate system
    * `gaze`
    * `recording`
    * `subject_cid`

Ground truth NPY updated by Leo on 250711:
    * `subject_cid`: updated
    * `IOU_L`
    * `IOU_R`
    * `headpose`
    also add new PATH_VALID_ID.
'''
import os
import cv2
import torch
import random
import numpy as np

from data.augmenter.augmenter import initialize_augmenter
import data.consts as consts

from torch.utils.data import Dataset as D

from data.gaze360.gaze360_2d_sampler import Gaze360_2D_Sampler

ROOT = '/data/gaze360/'

PATH_GT = 'gt_gaze360_with_conf_v4.npy' # new gt file by Leo
PATH_VALID_ID = 'gaze360_valid_ids.npy' # a list of valid ids by Leo

DEFAULT_AUG_ARGS_PATH = 'config/augmenter/gaze360_train.yaml' # a copy of gc3d augmenter train yaml

_DIRS = {
    'face': 'face_224',
    'facemap': 'facemap',
    'facemap_v3': 'seg',
    'facemask': None,
    'glasses_landmarks': None,
}

class Dataset_Gaze360(D):
    def __init__(self, root=ROOT, path_gt=PATH_GT,
            split='train', debug=False, augment=True,
            inputs = ['face'], labels = ['gaze'], 
            rgb_no_normalization = False, return_tensor = False,
            flip_mode = 'off',
            img_channel_chw=True, 
            face_apply_mask = False, face_simulate_glasses = False,
            augmenter_dir=consts.AUG_DIR_MATERIALS, augmenter_cfg_path=None,
            augmenter_options:dict={},
            enable_w_wrt_subject_id=False,
            dataset_base=None,
            bin_sampler=False,
            bin_args={},
            target_N = None,
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

        Supported inputs & labels fields:
        face, gaze, headpose
        TODO: facemap, wE_L/R, wI_L/R
        '''
        if path_gt is None:
            path_gt = PATH_GT
        if dataset_base is None:
            path_npy = os.path.join(root, path_gt)
            self.gt_data = np.load(path_npy, allow_pickle=True).item()
        else:
            print("[Gaze360]Loading GT from an existing Dataset_Gaze360 instance ...")
            self.gt_data = dataset_base.gt_data

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
        self.child_of_hybrid = child_of_hybrid
        
        self.valid_ids = self._filter_ids()
        self.id_list = self.valid_ids

        self.bin = bin_sampler
        if self.bin:
            self.nbins_pitch = bin_args['pitch_nbins']
            self.nbins_yaw = bin_args['yaw_nbins']
            self.pitch_bias = bin_args['pitch_bias']
            self.bin_size = bin_args.get('bin_size', 4)
            
            gaze = np.array(self.gt_data['gaze'])
            pitch_min = self.pitch_bias - self.nbins_pitch * self.bin_size / 2
            self.pitch_cid = np.clip((gaze[:, 0] * 180/np.pi - pitch_min) // self.bin_size, 0, self.nbins_pitch - 1).astype(int)

            yaw_min = 0 - self.nbins_yaw * self.bin_size / 2
            self.yaw_cid = np.clip((gaze[:, 1] * 180/np.pi - yaw_min) // self.bin_size, 0, self.nbins_yaw - 1).astype(int)

            self.sampler = Gaze360_2D_Sampler(self.gt_data, self.valid_ids,
                                            split=split, 
                                            bin_args=bin_args,
                                            pitch_cid=self.pitch_cid, yaw_cid=self.yaw_cid,
                                            target_N=target_N)
            self.id_list = self.sampler.sample()
            self.num_data = len(self.id_list)
            self.target_N = target_N # should be identical or close to self.num_data

            self.limit_pitch_u = (self.pitch_bias + self.nbins_pitch / 2 * self.bin_size) * np.pi / 180
            self.limit_pitch_l = (self.pitch_bias - self.nbins_pitch / 2 * self.bin_size) * np.pi / 180
            self.limit_yaw_u = (self.nbins_yaw / 2) * self.bin_size * np.pi / 180
            self.limit_yaw_l = (-self.nbins_yaw / 2) * self.bin_size * np.pi / 180

        self.face_apply_mask = face_apply_mask and self.dirs['facemask'] is not None
        self.face_simulate_glasses = face_simulate_glasses and self.dirs['glasses_landmarks'] is not None and self.split=='train'

        self.augment = augment and self.split=='train'
        self.augment_args = augmenter_options
        if augmenter_cfg_path is None and self.split=='train':
            augmenter_cfg_path = DEFAULT_AUG_ARGS_PATH
        self.aug = initialize_augmenter(self.split, augmenter_dir, augmenter_cfg_path)
 
    def __len__(self):
        if self.debug:
            return 512
        return self.num_data

    def _filter_ids(self):

        valid_ids = np.load(os.path.join(self.root, PATH_VALID_ID), allow_pickle=True) # now the npy file saves valid ids
        if self.split == 'train':
            valid_ids = [i for i in valid_ids if self.gt_data['split'][i] == 'train']
        else:
            valid_ids = [i for i in valid_ids if self.gt_data['split'][i] == 'test']
        self.num_data = len(valid_ids)
        return valid_ids

    def _load_rgb(self, type:str, i:int):
        '''
        type in ("face", "facemap", "facemap_v3")
        We have face images only at the moment.
        '''
        path = os.path.join(self.root, self.dirs[type], self.gt_data['fn'][i]+'.png')
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

        img_dict = {}
        for cat in consts.CATEGORIES:
            img_dict[cat] = {}
        for name in self.all:
            if (name in (consts.ANGULAR_TYPES)) or (name == 'w_wrt_subject_id'):
                val = self.gt_data[name][index]
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
                    subject_cid += 10000 # 0-1092
                values[name] = subject_cid
            elif name in consts.CONST_TYPES or name.startswith('w'):
                # if name == 'subject_cid':
                #     subject_cid = self.gt_data['new_subject_cid'][index]
                #     val = '_'.join([subject_cid[0], str(subject_cid[1])])
                #     values[name] = val
                if name == 'id':
                    val = index
                    values[name] = val
                else:
                    val = self.gt_data[name][index]
                    values[name] = val
            else: #'face'
                cat = consts.get_category(name)
                img_dict[cat][name] = self._load_rgb(name, index)

        img_mask = None
        # # face mask not supported yet..
        # if self.augment and self.face_apply_mask and 'face' in self.all:
        #     # load face mask
        #     mask_path = os.path.join(self.root, self.dirs['facemask'], str(id_eq)+'.png')
        #     if os.path.isfile(mask_path):
        #         img_mask = cv2.imread(mask_path)

        # batch-augment
        if not self.rgb_no_norm:
            if self.augment:
                for cat in consts.CATEGORIES:
                    if len(img_dict[cat]) == 0: continue
                    if cat=='face':
                        glasses_landmarks = None
                        # # glasses simulation not supported yet..
                        # if self.face_simulate_glasses:
                        #     path_gs = os.path.join(self.root, self.dirs['glasses_landmarks'], str(id_eq)+'.npy')
                        #     glasses_landmarks = np.load(path_gs) if os.path.isfile(path_gs) else None
                    img_dict[cat] = self.aug.augment(None,
                                                add_mask=self.face_apply_mask, rgb_mask=img_mask,
                                                img_dict=img_dict[cat], glasses_landmarks=glasses_landmarks,
                                                flip=flip,
                                                **self.augment_args)
            else:
                for cat in consts.CATEGORIES:
                    if len(img_dict[cat]) == 0: continue
                    img_dict[cat] = self.aug.augment(None,
                                                    img_dict=img_dict[cat],
                                                    flip=flip,
                                                    **self.augment_args)

        # transpose & unpack img_dict => values
        for cat in consts.CATEGORIES:
            if self.chw:  # HWC (TF, numpy) => CHW (PyTorch)
                for key, rgb in img_dict[cat].items():
                    if getattr(rgb, 'ndim', 1) == 3:
                        img_dict[cat][key] = rgb.transpose((2, 0, 1))
            values.update(img_dict[cat])

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

if __name__ == "__main__":
    d = Dataset_Gaze360()
    Xs, labels = d.__getitem__(2)