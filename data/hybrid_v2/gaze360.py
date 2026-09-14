'''
Gaze360 dataset - V2 (Dataset_Gaze360_V2).

Refactored for Dataset_Hybrid_V2: the sampler is gone. This class only loads and
filters; it exposes valid_ids + pitch_cid/yaw_cid (on the hybrid grid) and a
_get(gt_idx) that holds all the augment/flip loading logic. The hybrid does all
sampling. See hybrid_v2/base.py for the shared boilerplate.

Ported from data/dataset/gaze360.py (Dataset_Gaze360), dropping bin_sampler /
target_N / Gaze360_2D_Sampler. Modeled structurally on hybrid_v2/gazegene.py.

GT NPY keys (gt_gaze360_with_conf_v4.npy): fn, split, gaze, subject_cid,
headpose, IOU_L, IOU_R, recording, gaze_ecs.
'''
import os
import cv2
import torch
import random
import numpy as np

from data.augmenter.augmenter import initialize_augmenter, process_getitem_values
import data.consts as consts

from data.hybrid_v2.base import HybridGridDataset

ROOT = '/data/gaze360/'

PATH_GT          = 'gt_gaze360_with_conf_v4.npy'   # new gt file by Leo
PATH_VALID_ID    = 'gaze360_valid_ids.npy'         # a list of valid ids by Leo
PATH_SUBJECT_LMK = None                            # no glasses-landmark support

DEFAULT_AUG_ARGS_PATH = 'config/augmenter/gaze360_train.yaml'  # copy of gc3d train yaml

_DIRS = {
    'face': 'face_224',
    'facemap': 'facemap',
    'facemap_v3': 'seg',
    'facemask': None,
    'greenmask': None,
}


class Dataset_Gaze360_V2(HybridGridDataset):
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
            bin_args={},
            skip_val=False,
            child_of_hybrid=False,
            **kwargs):
        '''
        Same loading / augment behaviour as the original Dataset_Gaze360.
        Removed (vs. the original): bin_sampler, target_N, self.sampler.

        bin_args defines the hybrid grid and is REQUIRED:
            {'pitch_nbins', 'yaw_nbins', 'pitch_bias', ['bin_size']}.
        inputs/labels must be a superset of what the hybrid will read from this
        sub (the hybrid injects 'dataset_id'/'bin' itself).

        child_of_hybrid: kept only to offset subject_cid (+10000) so subject ids
            don't collide with the other datasets when used inside a hybrid.

        skip_val: accepted for parity with the other V2 subs. Gaze360 valid_ids
            are already split-encoded via gt_data['split'], so it is unused here.

        Supported inputs & labels fields:
        face, gaze, headpose, facemap, subject_cid, bin
        Not supported (reference has none): mixup, eye crops, glasses/mask synth.
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

        self.update_aug_states()

        if dataset_base is None:
            path_npy = os.path.join(root, path_gt)
            self.gt_data = np.load(path_npy, allow_pickle=True).item()
        else:
            print("[Gaze360]Loaded GT from an existing Dataset_Gaze360 instance.")
            self.gt_data = dataset_base.gt_data

        self.inputs = inputs
        self.labels = labels
        self.enable_w_wrt_subject_id = enable_w_wrt_subject_id
        if self.enable_w_wrt_subject_id:
            self.labels += ['w_wrt_subject_id']

        self.all = inputs+labels

        self.flip_mode = flip_mode if self.split=='train' else 'off'
        self.chw = img_channel_chw
        self.rgb_no_norm = rgb_no_normalization
        self.return_tensor = return_tensor
        self.child_of_hybrid = child_of_hybrid
        self.skip_val = skip_val

        # bins + valid_ids on the hybrid grid (replaces the old bin/sampler block)
        self._setup_bins_and_valid_ids(bin_args, self.gt_data['gaze'], self.gt_data.get('headpose'))

    # ===================================================================
    # Dataset-specific filtering (the original split-based id selection)
    # ===================================================================
    def _filter_dataset_specific(self):
        valid_ids = np.load(os.path.join(self.root, PATH_VALID_ID), allow_pickle=True)
        key = 'train' if self.split == 'train' else 'test'
        return [int(i) for i in valid_ids if self.gt_data['split'][i] == key]

    def _load_rgb(self, type:str, i:int):
        '''
        type in ("face", "facemap", "facemap_v3")
        Gaze360 paths are FILENAME-based (gt_data['fn']), not index-based.
        '''
        path = os.path.join(self.root, self.dirs[type], self.gt_data['fn'][i] + '.png')
        if not os.path.isfile(path):
            print("[Dataset]Warning: image {} not exists. Return None.".format(path))
            return None
        rgb = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if type in consts.RGB_TYPES:
            rgb = rgb[:,:,::-1]

        return rgb

    def _get(self, gt_idx, return_dict=False, force_flip=None):
        '''Core of the original __getitem__. `gt_idx` is a gt index (the hybrid
        and the inherited __getitem__ both feed gt indices in directly).

        force_flip: None -> decide flip from self.flip_mode (original behaviour).
                    True/False -> pin flip for this call (used by the hybrid's
                    n_views flip-views path so flip is deterministic per view).'''
        index = gt_idx
        values = {}
        id_eq = self.gt_data['fn'][index]

        # global random values
        flip = False
        if self.split=='train':
            if self.flip_mode=='on' or (self.flip_mode=='random' and random.random()<0.5):
                flip = True
        if force_flip is not None:   # hybrid views: pin flip deterministically per view
            flip = bool(force_flip)

        img_dicts = {}
        for cat in consts.CATEGORIES:
            img_dicts[cat] = {}
        for name in self.all:
            if (name in (consts.ANGULAR_TYPES)) or (name == 'w_wrt_subject_id'):
                val = np.array(self.gt_data[name][index])
                val = val.astype(np.float32)
                values[name] = val
            elif name == 'dataset_id':
                values[name] = self.dataset_id
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
                    subject_cid += 10000  # Gaze360 offset (reference); 0-1092 -> 10000+
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

    def update_aug_states(self):
        # for Joe's updates. Call this whenever an self.aug.* is changed externally.
        self.augment_glasses = self.aug.args.glasses.switch and self.split=='train' and PATH_SUBJECT_LMK is not None
        self.face_apply_mask = self.aug.args.add_mask.switch and self.dirs['facemask'] is not None and self.split=='train'
        self.face_replace_green = self.aug.args.replace_green.switch and self.dirs['greenmask'] is not None and self.split=='train'


if __name__ == "__main__":
    BA = {'pitch_nbins': 15, 'yaw_nbins': 17, 'pitch_bias': -12, 'bin_size': 4}
    d = Dataset_Gaze360_V2(bin_args=BA)
    Xs, labels = d[2]
