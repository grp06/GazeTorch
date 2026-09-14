'''
GazeCapture (3D)
'''
import os
import numpy as np
import random
import cv2

from data.augmenter.augmenter import initialize_augmenter, process_getitem_values
from data.augmenter.transform2d import rotate_pitchyaw
from data.gazecapture.gc3d_sampler import GC3D_Sampler
import data.consts as consts

from torch.utils.data import Dataset as D

_DIRS = {
    'face': "face", 
    'eye': 'eye_{}',
    'eyemap': 'eye_{}_eyemap2',
    'irismap': 'eye_{}_irismap2',
    'facemap': 'facemap',
    'facemask': 'facemask',
    'subject_landmarks': "gs_landmarks",
    'facemap_v3': 'seg',
    'greenmask': 'mp_backgrounds'
}

# given by data/gazecapture/process.gt, split()
_SUBJECT_ID_LIST = {
    'train': 'train250.npy',
    'val': 'val250.npy',
}

ROOT = '/data/gazecapture_3d'

DEFAULT_AUG_ARGS_PATH = 'config/augmenter/gc3d_train.yaml'

PATH_GT_DEFAULT = 'gt_p17y17_with_conf_v3.npy'
PATH_GT_LITTLE = 'gt_little.npy'

PATH_MIX = 'mixup_table_dict.npy'
PATH_INVALID_ID = 'invalid_ids_v2.npy'

PATH_SUBJECT_LMK = 'lmk.npy'
PATH_CAM_NUM = 'gc3d_camera.npy'

# deprecated
# self.face_mean = [0.56194926, 0.42180755, 0.3804766 ]
# self.face_std = [0.20368925, 0.17896926, 0.16430061]
# self.eye_mean = [0.54674796, 0.37119748, 0.33541909]
# self.eye_std = [0.19691373, 0.15854966, 0.14152511]

class Dataset_GC3d(D):
    def __init__(self, root=ROOT, path_gt=None,
        split='train', debug=False, augment=True, 
        inputs = ['face'], labels = ['gaze'],
        sampler_do_shuffle=True,
        little_mode=False,
        population=500, no_sample=False,
        subjects=None, subjects_excluded=["01090"],
        rgb_no_normalization = False, return_tensor = False,
        eye_resize_320=False, eye_oneside_noise = 0,
        face_noise = True,
        bin_sampler = False, 
        flip_mode = 'off',
        img_channel_chw = True,
        augmenter_base=None,
        augmenter_dir=consts.AUG_DIR_MATERIALS, augmenter_cfg_path=None,
        augmenter_options:dict={},
        enable_w_wrt_subject_id = False,
        same_person_sample_num = 1,
        mixup=False, mixup_halfandhalf=False, mixup_thresh=0.0,
        skip_val=False,
        bin_args={}, # can contain: n_bins_pitch=17, n_bins_yaw=17, pitch_bias=-8, bin_size=4
        target_N=None,
        dataset_base=None,
        child_of_hybrid=False,
        **kwargs):
        '''
        split: train/val
        dir_* (**kwargs in the code): images location (can be face/eye/eyemap/irismap/facemap/facemask)
        debug: if True, the dataset has much less samples so as to run trainning epochs faster
        dir_*: images location (can be face/eye/eyemap/irismap/facemap/facemask)
        flip_mode: can be "off"(default)/"on"/"random". for training only. 
            * "off": never apply horizontal flip.
            * "on": always flip.
            * "random": 50% probability to flip.
        img_channel_chw: if True, all images are in C*H*W format (PyTorch default)
        return_tensor: if True, apply torch.from_numpy() so that output values are Tensors.
        augment: if True and split=='train', switch on data augmentation
        augmenter*: see data.augmenter.augmenter.initialize_augmenter()

        Supported inputs & labels fields:
        face, eye_L, eye_R, gaze, headpose # old NPYs may use "head_pose" but it's fixed in recent versions
        bin (pitch_cid, yaw_cid)
        fn, subject_cid
        gaze_L, gaze_R
        facemap, facemap_eyelid
        eyemap_L, eyemap_R, irismap_L, irismap_R
        wE_L/R, wI_L/R

        GC3D-exclusive attributes
        population: randomly sample ~ samples per subject
            * for split=='train' ONLY
        no_sample: always avoid sampling regardless of split
        subjects: if not None (a list of string IDs), only take samples of these subjects
        subjects_excluded: if not None, samples of these subjects are discarded
            * Subject # 1090 contains too many invalid data
        little_mode: if True, I am a minimalistic GC3D dataset with 2200 samples (220 subjects, 10 samples each).
            augmentation, ID filtering, sampling & shuffling mechanisms are disabled. Samples are in `fn` order.
        sampler_do_shuffle: True in default. If False, GC3D sampler doesn't shuffle in __init__().
            * However, the sampler will still shuffle if user invokes shuffle().
        skip_val: (for training) if True, include all samples including val samples

        EYE attributes
        * eye_resize_320: if True, resize from 150x90 to 320x192
        * eye_oneside_noise: [0,1], probability to attach heavy noise on one of 2 eye images.

        FACE attributes
        * face_apply_mask: if True, switch on "random facemask" in data aug.
        * face_noise: if True, randomly apply global noise.
        * face_replace_green: if True, replace the background with random solid color or an image (see data/gazegene/backgroundremover.py).

        BIN attributes
        * bin_sampler: if True,
            * apply a Sampler (from bins) so that the sampled data obeys normal distribution roughly
            * gaze values are clipped according to bin values (e.g. for 9bins, bias=0, bin_size=4, then a gaze label is clipped to [-18, +18]deg)
        * bin_limit: (deprecated) if True, gaze values are clipped according to bin values (e.g. for 9bins, bias=0, bin_size=4, then gaze is clipped to [-18, +18]deg)        
        * bin_args{}

        OTHER attributes
        * rgb_no_normalization: if True, dataset returns RGB images (e.g. face & eyes) without any normalization nor augmentation. Only use this in debug (e.g. to demo some images).
        * return_tensor: if True, pack every numpy ndarrays into Torch tensors.
        * enable_w_wrt_subject_id: if True, attach "w_wrt_subject_id" at the end of labels
            * REGARDLESS OF whether it's already in inputs/labels
        * dataset_base: default=None. Otherwise, it should be an existing Dataset_XGaze instance. The new instance will load `gt_data` from it.
        * child_of_hybrid: should be True when it's part of a hybrid dataset instance. Otherwise False.
            * modifies the subject_cid to avoid overlapping with other datasets.
        '''

        self.root = root
        self.little_mode = little_mode
        self.split = split
        self.debug = debug
        self.dirs = _DIRS

        if path_gt is None:
            if little_mode:
                path_gt = PATH_GT_LITTLE
            else:
                path_gt = PATH_GT_DEFAULT

        path_npy_train_sample = os.path.join(root, _SUBJECT_ID_LIST['train'])
        path_npy_val_sample = os.path.join(root, _SUBJECT_ID_LIST['val'])

        # Augmenter
        self.augment = augment and self.split=='train'
        self.augment_args = augmenter_options
        if augmenter_cfg_path is None and self.split=='train':
            augmenter_cfg_path = DEFAULT_AUG_ARGS_PATH
        augment_split = 'train' if self.augment else 'test'
        if augmenter_base is None:
            self.aug = initialize_augmenter(augment_split, augmenter_dir, augmenter_cfg_path)
        else:
            self.aug = augmenter_base

        # Joe (25.12.15) - Remove args for augmenter and only use augmenter config YAML file instead
        # LZH: wrap them into a new function.            
        self.update_aug_states()

        # mixup
        self.mixup = mixup and split=='train'
        self.mixup_halfandhalf = mixup_halfandhalf
        
        if dataset_base is None:
            path_npy = os.path.join(root, path_gt)
            self.gt_data = np.load(path_npy, allow_pickle=True).item()
            if self.mixup:
                self.mixup_table = np.load(os.path.join(root, PATH_MIX), allow_pickle=True).item()
                self.mixup_thresh = mixup_thresh
                print("[GC3D] Mixup data Loaded.")
            if self.augment_glasses:
                self.subject_landmarks = np.load(os.path.join(root, PATH_SUBJECT_LMK), allow_pickle=True)
                self.subject_headposes = self.gt_data['headpose']
                print("[GC3D] Subject Landmarks (for glasses synth) Loaded.")
        else:
            # load from an existing dataset instance
            self.gt_data = dataset_base.gt_data
            if self.mixup:
                self.mixup_table = dataset_base.mixup_table
                self.mixup_thresh = dataset_base.mixup_thresh
            if self.augment_glasses:
                self.subject_landmarks = dataset_base.subject_landmarks
                self.subject_headposes = dataset_base.subject_headposes
            print("[GC3D]Loaded GT from an existing Dataset_GC3d instance.")

        self.inputs = inputs
        if enable_w_wrt_subject_id:
            labels.append('w_wrt_subject_id')
        self.labels = labels
        self.all = inputs + labels
        
        for k, v in kwargs.items():
            if k.startswith("dir_"):
                name = k[4:]
                self.dirs[name] = v
        # fill "L", "R"
        for name in ("eye", "eyemap", "irismap"):
            for side in ("L", "R"):
                k = name + "_" + side
                self.dirs[k] = self.dirs[name].format(side)
        
        self.num_data_o = len(self.gt_data['fn'])
        self.id_list_o = list(range(self.num_data_o))

        self.eye_resize_320 = eye_resize_320
        self.eye_oneside_noise = eye_oneside_noise and self.split=='train'
        self.face_noise = face_noise and self.split=='train'
        self.flip_mode = flip_mode if self.split=='train' else 'off'
        self.chw = img_channel_chw
        self.bin = bin_sampler
        self.rgb_no_norm = rgb_no_normalization
        self.return_tensor = return_tensor
        self.child_of_hybrid = child_of_hybrid
        # if self.bin:
        if len(bin_args)==0:
            self.pitch_cid = self.gt_data['pitch_cid']
            self.yaw_cid = self.gt_data['yaw_cid']

            self.nbins_pitch = self.gt_data['pitch_nbins']
            self.nbins_yaw = self.gt_data['yaw_nbins']
            self.pitch_bias = self.gt_data['pitch_bias']
            self.bin_size = self.gt_data["bin_size"]
        else:
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

        if little_mode:
            self.id_list = self.id_list_o
        else:
            if subjects is not None:
                self.subjects = subjects
            elif self.split=='train' and skip_val:
                print("[GC3D]Warning: skip_val = True. Will use val data for training!")
                self.subjects = None
            else:
                self.gt_sample_list = np.load(path_npy_train_sample, allow_pickle=True).item() if self.split == 'train' else np.load(path_npy_val_sample, allow_pickle=True).item() # participants training gt's
                self.subjects = self.gt_sample_list['id'] # list of subjects (5-digit str)
            self.subjects_ex = subjects_excluded
            self.valid_ids, self.D_roster, self.D_pop = self._filter_ids()

            self.n_per_person = population if no_sample else 0
            self.same_person_sample_num = same_person_sample_num

            if self.bin:
                self.sampler = GC3D_Sampler(self.valid_ids, self.D_roster, self.gt_data,
                                            n_per_person=self.n_per_person,
                                            split=split,
                                            yaw_cid=self.yaw_cid,
                                            bin_args=bin_args, target_N=target_N)
                self.id_list = self.sampler.sample(self.same_person_sample_num, sampler_do_shuffle)
                self.target_N = target_N # should be identical or close to self.num_data
            else:
                self.id_list = self.valid_ids

    def __len__(self):
        if self.debug:
            return 1024
        return len(self.id_list)

    def _filter_ids(self):
        # Jan 26 '23 udpate: form a new "roster", recording the valid id list per subject
        valid_ids = []
        roster = {}
        population = {}
        invalid_ids = np.load(os.path.join(self.root, PATH_INVALID_ID), allow_pickle=True)
        for i in self.id_list_o:
            if i in invalid_ids:
                continue

            valid_flag = True
            fn = self.gt_data['fn'][i]
            subject = fn[:5]
            if self.subjects is not None and not subject in self.subjects: continue
            if self.subjects_ex is not None and subject in self.subjects_ex: continue
            for name in consts.IMG_TYPES:
                if name not in self.all: continue
                ext = '.png' if name in consts.PNG_TYPES else ".jpg"
                path = os.path.join(self.root, self.dirs[name], fn+ext)
                if not os.path.isfile(path):
                    valid_flag = False
                    break
            if valid_flag:
                if subject not in roster:
                    roster[subject] = []
                    population[subject] = 0
                valid_ids.append(i)
                roster[subject].append(i)
                population[subject] += 1
        self.num_data = len(valid_ids)
        print("[Dataset]Valid ID list generated. # =", self.num_data)
        return valid_ids, roster, population

    def _load_rgb(self, type:str, i:str, mixup_id=None, alpha=0.0, **kwargs):
        '''
        type in ("face", "eye_L/R", "eyemap_L/R", "irismap_L/R", "facemap", "facemap_v3")
        for GC3D specially, `i` is a str corresponding to fn[index].
        '''
        ext = ".png" if type in consts.PNG_TYPES else ".jpg"

        if mixup_id is None:
            path = os.path.join(self.root, self.dirs[type], str(i)+ext)
            if not os.path.isfile(path):
                print("[Dataset]Warning: image {} not exists. Return None.".format(path))
                return None
            rgb = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        else:
            path_id = os.path.join(self.root, self.dirs[type], str(i)+ext)
            if not os.path.isfile(path_id):
                print("[Dataset]Warning: image {} not exists. Return None.".format(path_id))
                return None
            path_mixup_target = os.path.join(self.root, self.dirs[type], str(mixup_id)+ext)
            if not os.path.isfile(path_mixup_target): # double check mixup target image
                print("[Dataset]Warning: Mixup target image {} not exists. Return None.".format(path_mixup_target))
                return None

            rgb_id1 = cv2.imread(path_id, cv2.IMREAD_UNCHANGED)
            rgb_mixup_target = cv2.imread(path_mixup_target, cv2.IMREAD_UNCHANGED)

            rgb = cv2.addWeighted(rgb_id1, alpha, rgb_mixup_target, (1 - alpha), 0.0)

        if type in consts.RGB_TYPES:
            rgb = rgb[:,:,::-1]

        if type.startswith("eye_") and self.eye_resize_320:
            rgb = cv2.resize(rgb, (320, 192))

        if type.startswith('irismap') or type.startswith('eyemap'):
            if rgb.ndim > 2:
                rgb = rgb[..., 0]
            rgb = rgb[..., None] # HxWx1
            return rgb

        return rgb

    def __getitem__(self, i, return_dict=False): 
        index = self.id_list[i]
        index = int(index)
        id_eq = self.gt_data['fn'][index]
        
        values = {}

        if self.mixup_halfandhalf:
            alpha = 0.5
        else: alpha = 0.25 + random.random()*0.5 # [0.25, 0.75]

        mixup_valid_id = None
        mixup_valid_fn = None
        do_mixup = False

        if self.mixup:
            if len(self.mixup_table[index]) > 0 and random.random() >= self.mixup_thresh:
                do_mixup = True
                mixup_valid_id = int(random.choice(self.mixup_table[index]))
                mixup_valid_fn = self.gt_data['fn'][mixup_valid_id]

        # global random values
        oneside_noise_side = None
        flip = False
        if self.split=='train':
            if self.eye_oneside_noise>0 and random.random()<self.eye_oneside_noise:
                oneside_noise_side = 'L' if random.random()<0.5 else 'R'
            if self.flip_mode=='on' or (self.flip_mode=='random' and random.random()<0.5):
                flip = True

        # load each value involved
        img_dicts = {}
        for cat in consts.CATEGORIES:
            img_dicts[cat] = {}
        for name in self.all:
            if name in consts.ANGULAR_TYPES:
                if not do_mixup:
                    val = self.gt_data[name][index]
                    val = val.astype(np.float32)
                    values[name] = val
                else:
                    # mixup cases
                    val_img1 = self.gt_data[name][index]
                    val_img1 = val_img1.astype(np.float32)

                    val_img2 = self.gt_data[name][mixup_valid_id]
                    val_img2 = val_img2.astype(np.float32)

                    final_val = val_img1 * alpha + val_img2 * (1 - alpha)
                    values[name] = final_val
            elif name == 'id': # caution: sample ID won't support mixup (will just return original ID). for debug only.
                values_id = index
                if flip: values_id *= -1
                values[name] = values_id
            elif name == 'subject_cid':
                subject_cid = int(self.gt_data['subject_cid'][index])
                if self.child_of_hybrid:
                    subject_cid += 1000
                values[name] = subject_cid
            elif name in consts.CONST_TYPES or name.startswith('w'): # fields that don't change when flipping
                if not do_mixup:
                    val = self.gt_data[name][index]
                    values[name] = val
                else:
                    val_1 = self.gt_data[name][index]
                    val_2 = self.gt_data[name][mixup_valid_id]
                    values[name] = val_1 * alpha + val_2 * (1 - alpha)
            elif name=="bin":
                if not do_mixup:
                    pitch_bin_id = self.pitch_cid[index]
                    yaw_bin_id = self.yaw_cid[index]
                    if flip:
                        yaw_bin_id = -yaw_bin_id + self.nbins_yaw - 1
                    values[name] = np.array([pitch_bin_id, yaw_bin_id])
                else:
                    # re-calculate bin value
                    val_img1 = self.gt_data['gaze'][index]
                    val_img1 = val_img1.astype(np.float32)

                    val_img2 = self.gt_data['gaze'][mixup_valid_id]
                    val_img2 = val_img2.astype(np.float32)

                    gaze_val_combine = val_img1 * alpha + val_img2 * (1 - alpha)

                    nbins_pitch = self.nbins_pitch
                    nbins_yaw = self.nbins_yaw
                    pitch_bias = self.pitch_bias
                    bin_size = self.bin_size

                    pitch_bin_id = int(np.clip((((gaze_val_combine[0] * 180 / np.pi) - pitch_bias) // bin_size) + (nbins_pitch // 2 + 0.5), 0, nbins_pitch-1))
                    yaw_bin_id = int(np.clip(((gaze_val_combine[1] * 180 / np.pi) // bin_size) + (nbins_yaw // 2 + 0.5), 0, nbins_yaw-1))

                    if flip:
                        yaw_bin_id = -yaw_bin_id + nbins_yaw - 1

                    values[name] = np.array([pitch_bin_id, yaw_bin_id])
            elif 'aug' in name:
                pass
            else: #image
                # global_noise_std = 0
                # if name in ('eye_L', 'eye_R'):
                #     global_noise_std = 0.25 if oneside_noise_side == name[-1] else 0
                # elif name=='face':
                #     if self.face_noise and random.random()<0.5:
                #         global_noise_std = 0.25 * random.random()
                if not do_mixup:
                    cat = consts.get_category(name)
                    img_dicts[cat][name] = self._load_rgb(name, id_eq)
                else:
                    cat = consts.get_category(name)
                    img_dicts[cat][name] = self._load_rgb(name, id_eq, mixup_valid_fn, alpha)

        img_mask = None
        if self.augment and self.face_apply_mask and 'face' in self.all:
            # load face mask
            mask_path = os.path.join(self.root, self.dirs['facemask'], str(id_eq)+'.png')
            if os.path.isfile(mask_path):
                img_mask = cv2.imread(mask_path)
                
        
        # Background green mask
        green_mask = None
        if self.augment and self.face_replace_green and 'face' in self.all:
            # load background mask
            green_mask_path = os.path.join(self.root, self.dirs['greenmask'], str(id_eq)+'.jpg')
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

                    # gc3d find closest cam id then pass it to augmenter:
                    # sample_cam_id = self.camera_check[index]
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
                                                      **self.augment_args)

        # transpose & unpack img_dict => values
        for cat in consts.CATEGORIES:
            if self.chw:  # HWC (TF, numpy) => CHW (PyTorch)
                for key, rgb in img_dicts[cat].items():
                    if getattr(rgb, 'ndim', 1) == 3:
                        img_dicts[cat][key] = rgb.transpose((2, 0, 1))
            values.update(img_dicts[cat])

        # if image is rotated in augmentation, gaze should be rotated too
            # NOTE: rotation may cause problems in head pose angle. not recommended to apply rotation if `headpose` in `self.all`.
        if self.aug.t2d.buffer_angle != 0 and "gaze" in values:
            g = values['gaze']
            values['gaze'] = rotate_pitchyaw(g, self.aug.t2d.buffer_angle)

        # if flip=True:
            # revert yaw angles.
            # Switch L/R.
            # WARNING: L, R fields must be in inputs/labels BOTH.
        if flip:
            for name in self.all:
                if name in consts.ANGULAR_TYPES:
                    values[name][1] = -values[name][1] # revert yaw value
            for name in list(consts.SINGLE_EYE_TYPES) + ['gaze']:
                if name+'_L' in self.all:
                    v_L = values[name+'_L']
                    v_R = values[name+'_R']
                    values[name+'_R'] = v_L
                    values[name+'_L'] = v_R

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
        if self.little_mode: return
        self.id_list = self.sampler.sample(self.same_person_sample_num)
        random.shuffle(self.id_list)
        print("[Dataset]Shuffled.")

    def update_aug_states(self):
        # for Joe's updates. Call this whenever an self.aug.* is changed externally.
        self.augment_glasses = self.aug.args.glasses.switch and self.split=='train' and PATH_SUBJECT_LMK is not None
        self.face_apply_mask = self.aug.args.add_mask.switch and self.dirs['facemask'] is not None and self.split=='train'
        self.face_replace_green = self.aug.args.replace_green.switch and self.dirs['greenmask'] is not None and self.split=='train'