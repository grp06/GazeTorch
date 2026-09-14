'''
XGaze dataset class V2. Initial version on Jul 20 2022.
'''
import os
import numpy as np
import random
import cv2

from data.augmenter.augmenter import initialize_augmenter, process_getitem_values
from data.augmenter.transform2d import rotate_pitchyaw
from data.xgaze.xgaze_2d_sampler import XGaze_2D_Sampler
import data.consts as consts

from torch.utils.data import Dataset as D

ROOT = '/data/xgaze/448'

PATH_GT = "gt_p17y17b16_with_confs_v3.npy"
PATH_MIX = 'mixup_cam013456_iou80D.npy'
_DIRS = {
    'face': "face_224", 
    'eye': 'eye_{}_0610',
    'eyemap': 'eye_{}_eyemapP',
    'irismap': 'eye_{}_irismap0.2',
    'facemap': 'facemap_224v2',
    'facemask': 'facemask_224',
    'greenmask': 'green',
    'subject_landmarks': "gs_landmarks",
    'facemap_v3': 'seg',
}

PATH_GT_TEST = 'gt_face_224_test_with_conf_v3.npy'
_DIRS_TEST = {
    'face': "face_224_test", 
    # 'eye': 'eye_{}_0610',
    # 'eyemap': 'eye_{}_eyemapP',
    # 'irismap': 'eye_{}_irismap0.2',
    'facemap': 'facemap_224_test',
    'facemask': None, # 'facemask_224',
    'greenmask': None, # 'green',
    'subject_landmarks': None, #"gs_landmarks",
    'facemap_v3': 'seg_test',
}

PATH_SUBJECT_LMK = 'lmk_224.npy'

class Dataset_XGaze(D):
    def __init__(self, root=ROOT, path_gt=None, split='train', augment=False, debug=False, 
            inputs = ['face'], labels = ['gaze'],
            rgb_no_normalization = False, return_tensor=False,
            eye_resize_320=False, eye_oneside_noise = 0,
            face_noise = True, 
            bin_sampler = False, 
            flip_mode = 'off',
            cam_id=None, skip_val=False, 
            img_channel_chw=True,
            augmenter_base = None,
            augmenter_dir=consts.AUG_DIR_MATERIALS, augmenter_cfg_path=None,
            augmenter_options:dict={},
            mixup=False, mixup_halfandhalf=False, mixup_thresh=0.0,
            bin_args={}, # can contain: pitch_nbins=17, yaw_nbins=17, pitch_bias=-8, bin_size=4
            target_N = None,
            dataset_base = None,
            child_of_hybrid = False,
            n_views = 1,
            **kwargs):
        '''
        root: dataset root dir
        path_npy: ground truth location
        split: train/val
            * Dec 2022 update: add "test" split.
        dir_* (**kwargs in the code): images location (can be face/eye/eyemap/irismap/facemap/facemask)
        flip_mode: can be "off"(default)/"on"/"random". for training only. 
            * "off": never apply horizontal flip.
            * "on": always flip.
            * "random": 50% probability to flip.
        debug: if True, the dataset only contains 512 samples
        img_channel_chw: if True, all images are in C*H*W format (PyTorch default)
        return_tensor: if True, apply torch.from_numpy() so that output values are Tensors.
        augmenter*: see data.augmenter.augmenter.initialize_augmenter()
        mixup: if True, enable mixup features
            * mixup_thresh: probability not to use mixup
        bin_args: the bin settings, if specified
            * when specified, sampler is enabled too (self.bin=True)
        target_N: if not None, # of samples each time sampler draws. Determines len().
            * if None, sampler will follow a Gaussian distribution (deprecated)

        Suported input & output fields:
        face, eye_L, eye_R, gaze, headpose
        cam_index, frame_index, subject
        pitch_cid, yaw_cid, pitch_nbins, yaw_nbins, bin_size, pitch_bias
        id, cam_id, subject_cid
        eye_L_avail, eye_R_avail
        facemap, facemap_v3
        eyemap_L, eyemap_R, irismap_L, irismap_R
        wE_L/R, wI_L/R

        XGaze-exclusive attributes
        cam_id: None/list. specifies a subset of camera to use
        skip_val: (for training) if True, include all samples including val samples

        EYE attributes
        * eye_resize_320: if True, resize from 150x90 to 320x192
        * eye_oneside_noise: [0,1], probability to attach heavy noise on one of 2 eye images.

        FACE attributes
        * face_apply_mask: if True, switch on "random facemask" in data aug (only when self.augment=True).
        * face_noise: if True, randomly apply global noise.
        * face_replace_green: if True, replace the green background with random solid color or an image (see Green_Background_Replacer class).
            * NOTE: applicable for XGaze only.
        * face_simulate_glasses: if True, simulate glasses reflection via Augmenter.Patcher

        BIN attributes
        * bin_sampler: if True,
            * apply a Sampler (from bins) so that the sampled data obeys normal distribution roughly
            * gaze values are clipped according to bin values (e.g. for 9bins, bias=0, bin_size=4, then a gaze label is clipped to [-18, +18]deg)
        * bin_args{}

        OTHER attributes
        * rgb_no_normalization: if True, dataset returns RGB images (e.g. face & eyes) without any normalization nor augmentation. Only use this in debug (e.g. to demo some images).
        * return_tensor: if True, pack every numpy ndarrays into Torch tensors.
        * dataset_base: default=None. Otherwise, it should be an existing Dataset_XGaze instance. The new instance will load `gt_data` from it.
        * n_views: simulate multiple views. when using __getitem__(i), {i // n_views, ..., i // n_views + n_views - 1} corresponds to the same sample.
            * u MUST use sequential DataLoader, and switch off "jitter" (geometric jitter) in augmentation yaml.
            * can't be used in hybrid mode.

        '''
        self.split = split
        if split in ['train', 'val']:
            self.start_val = 672228
            self.num = 756540
            # self.num_samples = self.num//18
            # if skip_val:
            #     self.num_samples_train = self.num//18
            # else:
            #     self.num_samples_train = self.start_val//18
            # self.num_samples_val = self.num_samples - self.num_samples_train
            # self.num_per_sample = 18 if cam_id is None else len(cam_id)
            self.dirs = _DIRS
            self.skip_val = skip_val
            if path_gt is None:
                path_gt = PATH_GT
            if self.skip_val:
                print("[XGaze]Warning: skip_val = True. Will use val data for training!")
                self.id_list_o = list(range(self.num))
            else:
                self.id_list_o = list(range(self.start_val)) if self.split=='train' else list(range(self.start_val, self.num))
        else: # test mode
            self.num = 3200
            self.dirs = _DIRS_TEST
            self.id_list_o = list(range(self.num))
            if path_gt is None:
                path_gt = PATH_GT_TEST

        # Augmenter
        self.augment = augment and self.split=='train'
        augment_split = 'train' if self.augment else 'test'
        self.augment_args = augmenter_options
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

        # Load NPY files
        if dataset_base is None:
            path_npy = os.path.join(root, path_gt)
            self.gt_data = np.load(path_npy, allow_pickle=True).item()
            if self.mixup:
                self.mixup_table = np.load(os.path.join(root, PATH_MIX), allow_pickle=True).item()
                self.mixup_thresh = mixup_thresh
                print("[XGaze] Mixup data Loaded.")
            if self.augment_glasses:
                self.subject_landmarks = np.load(os.path.join(root, PATH_SUBJECT_LMK), allow_pickle=True)
                self.subject_headposes = self.gt_data['headpose']
                print("[XGaze] Subject Landmarks (for glasses synth) Loaded.")
        else:
            # load from an existing dataset instance
            self.gt_data = dataset_base.gt_data
            if self.mixup:
                self.mixup_table = dataset_base.mixup_table
                self.mixup_thresh = dataset_base.mixup_thresh
            if self.augment_glasses:
                self.subject_landmarks = dataset_base.subject_landmarks
                self.subject_headposes = dataset_base.subject_headposes
            print("[XGaze]Loaded GT from an existing Dataset_XGaze instance.")

        self.inputs = inputs
        self.labels = labels
        self.all = inputs + labels
        for k, v in kwargs.items():
            if k.startswith("dir_"):
                name = k[4:]
                self.dirs[name] = v
        # fill "L", "R"
        # WARNING: not supported when split=="test"
        if self.split != 'test':
            for name in ("eye", "eyemap", "irismap"):
                for side in ("L", "R"):
                    k = name + "_" + side
                    self.dirs[k] = self.dirs[name].format(side)

        self.root = root
        self.debug = debug
        self.cams = cam_id

        self.id_list_o = self._filter_ids()

        self.eye_resize_320 = eye_resize_320
        self.eye_oneside_noise = eye_oneside_noise and self.split=='train'

        self.face_noise = face_noise and self.split=='train'
        self.flip_mode = flip_mode if self.split=='train' else 'off'
        self.chw = img_channel_chw
        self.bin = bin_sampler
        self.rgb_no_norm = rgb_no_normalization
        self.return_tensor = return_tensor
        self.child_of_hybrid = child_of_hybrid
        self.n_views = n_views

        if len(bin_args)==0: # follow the bin setting in PATH_GT
            self.pitch_cid = self.gt_data['pitch_cid']
            self.yaw_cid = self.gt_data['yaw_cid']
            # bin_limit=True if bin_sampler=True
            if self.split=='test':
                self.nbins_pitch = self.gt_data['pitch_nbins'][0]
                self.nbins_yaw = self.gt_data['yaw_nbins'][0]
                self.pitch_bias = self.gt_data['pitch_bias'][0]
                self.bin_size = self.gt_data['bin_size'][0]
            else:
                self.nbins_pitch = self.gt_data['pitch_nbins']
                self.nbins_yaw = self.gt_data['yaw_nbins']
                self.pitch_bias = self.gt_data['pitch_bias']
                self.bin_size = self.gt_data['bin_size']
        else: # customized bin setting, given by bin_args
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
            self.sampler = XGaze_2D_Sampler(self.gt_data, self.id_list_o,
                                            split=split,
                                            cams=self.cams,
                                            side=None, 
                                            bin_args=bin_args,
                                            pitch_cid=self.pitch_cid, yaw_cid=self.yaw_cid,
                                            target_N=target_N)
            self.target_N = target_N # should be identical or close to self.num_data
            if self.n_views > 1:
                print("[XGaze]Internal multi-view mode ON!")
                self.shuffle()
            else:
                self.id_list = self.sampler.sample()
                self.num_data = len(self.id_list)
        else:
            self.id_list = self.id_list_o
            self.target_N = len(self.id_list_o) if target_N is None else target_N
            if self.split=='train':
                self.shuffle()
            print("[XGaze]Len of id_list:", len(self.id_list))
        
    def __len__(self):
        if self.debug:
            return 512
        return self.num_data

    def _filter_ids(self):
        id_list_new = []
        for i in self.id_list_o:
            valid_flag = True
            if self.cams is not None and i%18 not in self.cams:
                continue
            for name in consts.IMG_TYPES:
                if name not in self.all: continue
                ext = '.png' if name in consts.PNG_TYPES else ".jpg"
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
        '''
        type in ("face", "eye_L/R", "eyemap_L/R", "irismap_L/R", "facemap", "facemap_v3")
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

            rgb = cv2.addWeighted(rgb_id1, alpha, rgb_mixup_target, (1 - alpha), 0.0) # Linear blend

        if type in consts.RGB_TYPES:
            rgb = rgb[:,:,::-1] # Reverse because openCV stores in BGR instead of RGB

        if not type.startswith("face") and self.eye_resize_320:
            rgb = cv2.resize(rgb, (320, 192)) 

        if type.startswith('irismap') or type.startswith('eyemap'):
            if rgb.ndim > 2:
                rgb = rgb[..., 0]
            rgb = rgb[..., None] # HxWx1
            return rgb

        return rgb

    def __getitem__(self, i, return_dict=False): 
        index = self.id_list[i]
        id_eq = index
        
        values = {}

        if self.mixup_halfandhalf:
            alpha = 0.5
        else: alpha = 0.25 + random.random()*0.5 # [0.25, 0.75]

        mixup_valid_id = None
        do_mixup = False

        if self.mixup:
            mix_beta = random.random()
            if mix_beta >= self.mixup_thresh and len(self.mixup_table[index]) != 0:
                do_mixup = True
                mixup_valid_id = int(random.choice(self.mixup_table[index]))

        # global random values
        oneside_noise_side = None
        flip = False
        if self.split=='train':
            if self.eye_oneside_noise>0 and random.random()<self.eye_oneside_noise:
                oneside_noise_side = 'L' if random.random()<0.5 else 'R'
            if self.n_views > 1:
                flip = i%2 == 1
            elif self.flip_mode=='on' or (self.flip_mode=='random' and random.random()<0.5):
                flip = True

        # load each value involved
        img_dicts = {}
        for cat in consts.CATEGORIES:
            img_dicts[cat] = {}
        for name in self.all: # self.all <- input + labels 
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

                    pitch_bin_id = int(np.clip(
                        (((gaze_val_combine[0] * 180 / np.pi) - self.pitch_bias) // self.bin_size) + (self.nbins_pitch // 2 + 0.5),
                         0, self.nbins_pitch-1))
                    yaw_bin_id = int(np.clip(((gaze_val_combine[1] * 180 / np.pi) // self.bin_size) + (self.nbins_yaw // 2 + 0.5), 0, self.nbins_yaw-1))

                    if flip:
                        yaw_bin_id = -yaw_bin_id + self.nbins_yaw - 1

                    values[name] = np.array([pitch_bin_id, yaw_bin_id])
            elif 'aug' in name:
                pass
            else: #image
                if not do_mixup:
                    cat = consts.get_category(name)
                    img_dicts[cat][name] = self._load_rgb(name, id_eq)
                else:
                    cat = consts.get_category(name)
                    img_dicts[cat][name] = self._load_rgb(name, id_eq, mixup_valid_id, alpha)

        img_mask = None
        if self.augment and self.face_apply_mask and 'face' in self.all:
            # load face mask
            mask_path = os.path.join(self.root, self.dirs['facemask'], str(id_eq)+'.png')
            if os.path.isfile(mask_path):
                img_mask = cv2.imread(mask_path)

        green_mask = None
        if self.augment and self.face_replace_green and 'face' in self.all:
            # load green
            mask_path = os.path.join(self.root, self.dirs['greenmask'], str(id_eq)+'.jpg')
            if os.path.isfile(mask_path):
                green_mask = cv2.imread(mask_path)

        # batch-augment
        if not self.rgb_no_norm:
            if self.augment:
                for cat in consts.CATEGORIES:
                    if len(img_dicts[cat]) == 0: continue
                    subject_landmarks, subject_headpose = None, None
                    if cat=='face':
                        if self.augment_glasses:
                            subject_landmarks = self.subject_landmarks[id_eq]
                            subject_headpose = self.subject_headposes[id_eq]

                    augmented = self.aug.augment(None, 
                        add_mask=self.face_apply_mask, rgb_mask=img_mask,
                        replace_green=self.face_replace_green, rgb_green=green_mask,
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
            if self.chw: # HWC (TF, numpy) => CHW (PyTorch)
                for key, rgb in img_dicts[cat].items():
                    if getattr(rgb, 'ndim', 1) == 3:
                        img_dicts[cat][key] = rgb.transpose((2,0,1))
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

        # for "bin" case only, clip gaze value
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
        if self.bin:
            self.id_list = self.sampler.sample()
            if self.n_views > 1:
                id_list_repeated = []
                for i in self.id_list:
                    id_list_repeated += [i] * self.n_views
                    if len(id_list_repeated) >= self.target_N:
                        break
                self.id_list = id_list_repeated
        else:
            random.shuffle(self.id_list_o)
            self.id_list = self.id_list_o[:self.target_N]
        self.num_data = len(self.id_list)
        print("[Dataset]Shuffled.")

    def update_aug_states(self):
        # for Joe's updates. Call this whenever an self.aug.* is changed externally.
        self.augment_glasses = self.aug.args.glasses.switch and self.split=='train' and PATH_SUBJECT_LMK is not None
        self.face_apply_mask = self.aug.args.add_mask.switch and self.dirs['facemask'] is not None and self.split=='train'
        self.face_replace_green = self.aug.args.replace_green.switch and self.dirs['greenmask'] is not None and self.split=='train'


class Dataset_XGaze_singleeye(D):
    '''
    Single-eye mode (single_eye_mode=True):
        The dataset concatenates left & right eye-related values.
        For example, if the dataset has 1K face images, then len(this)=2K under this mode, with 1K left images and 1K right images.
        The remaining data (which is the same for L/R) will be repeated twice.
        Use "eye" instead of "eye_L", "eye_R" (same for eyemap, irismap) under this mode to switch on concatenation.
        
    '''
    def __init__(self, inputs=['eye'], labels=['gaze'], **kwargs):
        self.inputs = inputs
        self.labels = labels

        self.inputs_L = self.replace(self.inputs, 'L')
        self.labels_L = self.replace(self.labels, 'L')
        self.inputs_R = self.replace(self.inputs, 'R')
        self.labels_R = self.replace(self.labels, 'R')

        self.d_L = Dataset_XGaze(inputs=self.inputs_L, labels=self.labels_L, **kwargs)
        self.d_R = Dataset_XGaze(inputs=self.inputs_R, labels=self.labels_R, **kwargs)

        self.id_list = []
        self.id_list.extend(self.d_L.id_list)
        self.id_list.extend(self.d_R.id_list)

        self.len_L = len(self.d_L)
        self.len_R = len(self.d_R)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, i):
        side, id_eq = self.decode(i)
        return getattr(self, 'd_'+side)[id_eq]

    def decode(self, i):
        if i<self.len_L:
            return 'L', i
        else:
            return 'R', i-self.len_L
        
    @staticmethod
    def replace(L: list, side: str):
        # side must be 'L' or 'R'
        L_out = L.copy()
        for i, name in enumerate(L_out):
            if name in consts.SINGLE_EYE_TYPES:
                if name == 'gaze*':
                    L_out[i] = 'gaze_' + side
                else:
                    L_out[i] = name + '_' + side
        return L_out