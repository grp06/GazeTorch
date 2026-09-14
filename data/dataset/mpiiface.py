'''
by Zhenhao, Nov 1 2022
MPIIFaceGaze dataset class
rewritten to fit the new, unified Dataset definition style.
face only, not trident at the moment.
3000 images for each subject. so no sampling mechanism.

Apr 2023
mixup feature by Mahak & Zhenhao
May 2023
cross-validation feature by Zhenhao
'''
import os
import numpy as np
import random
import cv2

from data.augmenter.augmenter import initialize_augmenter
from data.augmenter.transform2d import rotate_pitchyaw
import data.consts as consts
from data.mixup import MixupProcessor
from data.mpiiface.mpii_2d_sampler import MPII_2D_Sampler

from torch.utils.data import Dataset as D

ROOT = '/data/mpiiface'
PATH_GT = 'gt_mpiiface_with_conf_v3.npy'
SUBJECTS = list(range(1, 16))
_DIRS = {
    'face': "224png", 
    'facemap': 'facemap',
    'facemap_v3': 'seg',
}
_MIXUP_DIR = "mixup_data"
_MIXUP_FILES = {
    'id': 'matrix.npy',
    'valid': 'binaryValue.npy',
}
CROSS_VALIDATION_RATIO = 0.02

VAL_LIST_PATH = "mpiiface_val.npy"

class Dataset_MPIIFace(D):
    def __init__(self, root=ROOT, path_gt=PATH_GT,
            split='train', debug=False, augment=True,
            inputs = ['face'], labels = ['gaze'], 
            val_subjects:list = [],
            rgb_no_normalization = False, return_tensor = False,
            flip_mode = 'off',
            persona_mode = False, persona_train = 'first', persona_val = 'rest', persona_n_calib = 100,
            mixup = False, mixup_halfandhalf=False, mixup_num_samples_cap=None,
            resample_num=None,
            img_channel_chw=True, 
            augmenter_dir=consts.AUG_DIR_MATERIALS, augmenter_cfg_path=None,
            augmenter_options:dict={},
            skip_every = 1,
            **kwargs):
        '''
        split: train/val
        debug: if True, the dataset has much less samples so as to run trainning epochs faster
        augment: if True and split=='train', switch on data augmentation
        dir_*: images location (can be face/eye/eyemap/irismap/facemap/facemask)
        flip_mode: can be "off"(default)/"on"/"random". For training only. 
            * "off": never apply horizontal flip.
            * "on": always flip.
            * "random": 50% probability to flip.
        img_channel_chw: if True, all images are in C*H*W format (PyTorch default)
        return_tensor: if True, apply torch.from_numpy() so that output values are Tensors.
        augmenter*: see data.augmenter.augmenter.initialize_augmenter()

        Supported inputs & labels fields:
        face, gaze, headpose
        facemap_v3, wE_L/R, wI_L/R
        landmarks (given by the dataset, 6 2D-landmark coordinates)
        subject_cid (#1-15)

        MPII-exclusive attributes
        val_subjects: the subject (#1-15) for validation.
            so, if split==train, the Dataset uses the complement subject IDs.
            If ==[], always apply all subjects.
        persona_mode: if True, then train/val split come from `val_subjects` only.
            * persona_n_calib = # of calib points, N
            * persona_train = "first"/"random"
                * "first": select the first N valid samples in self.id_list_o as training set
                * "random": pick N random samples as training set
            * persona_val = "all"/"rest"
                * "all": val set = all samples of `val_subjects`. For unlabeled setting.
                * "rest": val set = all samples except for training samples. For labeled setting.
        split=='train_cv'/'val_cv': May 2023 update
            * take a portion (15%) of "train" data as validation. For cross-validation setting.
        split=='val_gen': 2026 update
            * pick 200 samples per person as validation data. For domain generalization setting.
        mixup: Apr 2023 update
            * if True, generate online mixup images. For training only.
            * mixup_halfandhalf:
                * False (default): mixup ratio = rand()
                * True: mixup ratio = 0.5 always
            * mixup_num_samples_cap: if None, manually impose a cap to len(self). (debug sets it to 256)
                * For training only.
        resample_num: Jul 2023 update
            * if specified (int), generate a new self.id_list whose len = resample_num
                so that samples have uniform 2D distribution.
        '''
        
        path_npy = os.path.join(root, path_gt)
        self.gt_data = np.load(path_npy, allow_pickle=True).item()

        self.dirs = _DIRS
        self.inputs = inputs
        self.labels = labels
        self.all = inputs+labels

        for k, v in kwargs.items():
            if k.startswith("dir_"):
                name = k[4:]
                self.dirs[name] = v

        self.root = root
        self.split = split
        self.debug = debug

        self.mixup = mixup
        self.mixup_halfandhalf = mixup_halfandhalf
        if self.mixup:
            self.mixup_matrix = np.load(os.path.join(self.root, _MIXUP_DIR, _MIXUP_FILES['id'])).astype(int)
            self.mixup_valid = np.load(os.path.join(self.root, _MIXUP_DIR, _MIXUP_FILES['valid'])).astype(bool)

        self.num_data_o = len(self.gt_data['gaze'])
        is_training = split in ['train', 'train_cv', 'val_cv']
        if is_training and self.mixup:
            self.id_list_o = list(range(0, 45000*16))
        elif self.split == 'val_gen':
            path_id_list = os.path.join(root, VAL_LIST_PATH)
            self.id_list_o = np.load(path_id_list, allow_pickle=True)
        else:
            self.id_list_o = list(range(self.num_data_o))

        self.flip_mode = flip_mode if self.split=='train' else 'off'
        self.subject_ids = val_subjects
        self.chw = img_channel_chw
        self.rgb_no_norm = rgb_no_normalization
        self.return_tensor = return_tensor

        self.persona = persona_mode
        self.persona_train = persona_train
        self.persona_val = persona_val
        self.persona_n = persona_n_calib
        
        if self.persona:
            self.id_list = self._persona_ids(is_training)
        elif self.split == 'val_gen':
            self.id_list = self.id_list_o
        else:
            self.id_list = self._filter_ids(is_training)

        if split.endswith('cv'):
            np.random.seed(0)
            perm = np.random.permutation(self.id_list)
            n_sub = int(len(self.id_list) * CROSS_VALIDATION_RATIO)
            if split=='train_cv':
                self.id_list = perm[n_sub:]
            else: # val_cv
                self.id_list = perm[:n_sub]

        self.num_data = len(self.id_list)
        print("[Dataset]Valid ID list generated. # =", self.num_data)

        self.num_data_cap = mixup_num_samples_cap

        self.num_data_resample = resample_num
        if self.split=='train' and resample_num is not None:
            self.resample = True
            self.gt_valid_list = [i for i in self.id_list if i < 45000] # Zheng(23-08-03): bug fix for mixup case
            gaze_list = self.gt_data['gaze'][self.gt_valid_list]
            self.sampler = MPII_2D_Sampler(self.gt_valid_list, gaze_list, resample_num)
            self.num_data = resample_num
        else:
            self.resample = False

        self.augment = augment and self.split=='train'
        self.augment_args = augmenter_options
        self.aug = initialize_augmenter(self.split, augmenter_dir, augmenter_cfg_path)

        if 'train' not in self.split and skip_every > 1:
            print("[MPII] Skip every:", skip_every)
            self.id_list = self.id_list[::skip_every]
            self.num_data = len(self.id_list)

    def __len__(self):
        if self.debug:
            return 512
        if self.split=='train_cv' and self.num_data_cap is not None:
            return self.num_data_cap
        return self.num_data
    
    def _get_mixup_id(self, gt_id):
        subject_cids = []
        image_cids = []
        if gt_id < 45000:
            subject_id_num = self.gt_data['subject_cid'][gt_id]
            subject_cids.append(subject_id_num)
            subject_cids.append(None)
            image_cids.append(gt_id)
            image_cids.append(None)
        else:
            img1_num = gt_id % 45000
            subject_id_num = (gt_id // 45000) - 1
            if not self.mixup_valid[img1_num, subject_id_num]: # not a good mixup, e.g. IoU too small
                return [None, None], [None, None]
            img2_num = self.mixup_matrix[img1_num, subject_id_num]

            subject_cids.append((img1_num // 3000) + 1)
            subject_cids.append((img2_num // 3000) + 1)

            image_cids.append(img1_num)
            image_cids.append(img2_num)

        return subject_cids, image_cids

    def _filter_ids(self, exclude_subject_id=False):
        valid_ids = []
        ext = '.png'

        # mixup-exclusive variables
        self.mixup_id1 = []
        self.mixup_id2 = []

        if len(self.subject_ids) == 0:
            valid_subjects = SUBJECTS
        elif exclude_subject_id:
            valid_subjects = [f for f in SUBJECTS if f not in self.subject_ids] # complement of self.subject_ids
        else:
            valid_subjects = self.subject_ids
        
        for i in self.id_list_o:
            subject_cids, image_cids = self._get_mixup_id(i)
            self.mixup_id1.append(image_cids[0])
            self.mixup_id2.append(image_cids[1])
            valid_flag = True
            if subject_cids[0] is None:
                continue
            if subject_cids[1] is None:
                sid = subject_cids[0]
                if sid not in valid_subjects:
                    continue
                for name in consts.IMG_TYPES:
                    if name not in self.all: continue
                    path = os.path.join(self.root, self.dirs[name], str(image_cids[0])+ext) # if original image are JPGs, ext is more complex
                    if not os.path.isfile(path):
                        valid_flag = False
                        break
                if valid_flag:
                    valid_ids.append(i)
            else:
                # mixup case (len=2)
                if subject_cids[0] not in valid_subjects or subject_cids[1] not in valid_subjects:
                    continue
                for name in consts.IMG_TYPES:
                    if name not in self.all: continue
                    path_img1 = os.path.join(self.root, self.dirs[name], str(image_cids[0])+ext)
                    path_img2 = os.path.join(self.root, self.dirs[name], str(image_cids[1])+ext)
                    if not os.path.isfile(path_img1) or not os.path.isfile(path_img2):
                        valid_flag = False
                        break
                if valid_flag:
                    valid_ids.append(i)

        return valid_ids
    
    def _persona_ids(self, is_training=True):
        valid_ids = self._filter_ids(exclude_subject_id=False)
        if not is_training and self.persona_val=='all':
            return valid_ids

        ids = valid_ids.copy()
        if self.persona_train == 'random':
            ids = np.random.permutation(ids)
        train_ids = ids[:self.persona_n]
        if is_training:
            ids = train_ids
        elif self.persona_val == 'rest':
            ids = [i for i in valid_ids if i not in train_ids]
        else:
            raise Exception("[MPII]Invalid persona settings.")
        print("[Dataset]Persona mode ON.")
        return ids

    def _load_rgb(self, type:str, i:int, alpha=0.5, is_mixup=False):
        '''
        type in ("face", "facemap", "facemap_v3")
        We have face images only at the moment.
        
        Apr '23 update: support online mixup. `i` = ID in `id_list`; may not be aligned with GT.
            * mixup from self.mixup_id1/id2. 
        May update: if mixup & type==facemap_v3, then return a STACKED segmap (8-channeled)
        '''
        if not is_mixup: # no mixup
            path = os.path.join(self.root, self.dirs[type], '{}.png'.format(i))
            if not os.path.isfile(path):
                print("[Dataset]Warning: image {} not exists. Return None.".format(path))
                return None
            rgb = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        else:
            img1_path = os.path.join(self.root, self.dirs[type], '{}.png'.format(self.mixup_id1[i]))
            img2_path = os.path.join(self.root, self.dirs[type], '{}.png'.format(self.mixup_id2[i]))

            rgb = MixupProcessor.mix_images(img1_path, img2_path, alpha)

        if type in consts.RGB_TYPES:
            rgb = rgb[:,:,::-1]

        return rgb

    def __getitem__(self, i):     
        index = self.id_list[i]
        values = {}

        # mixup alpha
        if self.mixup_halfandhalf:
            alpha = 0.5
        else: alpha = random.random()

        if self.split == 'val_gen':
            is_mixup = False
        else:
            is_mixup = self.mixup_id2[index] is not None

        # global random values
        flip = False
        if self.split=='train':
            if self.flip_mode=='on' or (self.flip_mode=='random' and random.random()<0.5):
                flip = True

        img_dict = {}
        for name in self.all:
            if name in consts.ANGULAR_TYPES:
                if not is_mixup:
                    val = self.gt_data[name][index]
                    val = val.astype(np.float32)
                    values[name] = val
                else:
                    val_img1 = self.gt_data[name][self.mixup_id1[index]]
                    val_img1 = val_img1.astype(np.float32)

                    val_img2 = self.gt_data[name][self.mixup_id2[index]]
                    val_img2 = val_img2.astype(np.float32)

                    final_val = val_img1 * alpha + val_img2 * (1 - alpha)
                    values[name] = final_val
            elif name in consts.CONST_TYPES or name.startswith('w'): # fields that don't change when flipping
                if not is_mixup:
                    val = self.gt_data[name][index]
                    values[name] = val
                else:
                    values[name] = [self.gt_data[name][self.mixup_id1[index]],
                                    self.gt_data[name][self.mixup_id2[index]]]
            elif name == consts.ALPHA:# mixup-exclusive
                values[name] = alpha
            else: #'face'
                img_dict[name] = self._load_rgb(name, index, alpha, is_mixup)

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

        # if image is rotated in augmentation, gaze should be rotated too
        if self.aug.t2d.buffer_angle != 0 and "gaze" in values:
            g = values['gaze']
            values['gaze'] = rotate_pitchyaw(g, self.aug.t2d.buffer_angle)

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
        if self.resample:
            self.id_list = self.sampler.sample()
        else:
            self.id_list = np.random.permutation(self.id_list)