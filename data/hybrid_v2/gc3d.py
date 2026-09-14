'''
GazeCapture (3D) - V2 (Dataset_GC3d_V2).

Refactored for Dataset_Hybrid_V2: the sampler is gone. This class only loads and
filters; it exposes valid_ids + pitch_cid/yaw_cid (on the hybrid grid) and a
_get(gt_idx) holding all the augment/mixup/flip/eye loading logic. The hybrid
does all sampling. See hybrid_v2/base.py for the shared boilerplate.

Per-subject balancing (n_per_person), which used to live in GC3D_Sampler, now
runs inside _filter_dataset_specific(). SSL clique batching moved to the hybrid
layer (see dataset/hybrid_v2.py); the old 1D-yaw balancing is dropped (the 2D
bin-uniform sampling in the hybrid covers it).
'''
import os
import numpy as np
import random
import cv2

from data.augmenter.augmenter import initialize_augmenter, process_getitem_values
from data.augmenter.transform2d import rotate_pitchyaw
import data.consts as consts

from data.hybrid_v2.base import HybridGridDataset

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

PATH_MIX = 'mixup_table_dict.npy'
PATH_INVALID_ID = 'invalid_ids_v2.npy'

PATH_SUBJECT_LMK = 'lmk.npy'
PATH_CAM_NUM = 'gc3d_camera.npy'

class Dataset_GC3d_V2(HybridGridDataset):
    def __init__(self, root=ROOT, path_gt=None,
        split='train', debug=False, augment=True,
        inputs = ['face'], labels = ['gaze'],
        n_per_person=0,
        subjects=None, subjects_excluded=["01090"],
        rgb_no_normalization = False, return_tensor = False,
        eye_resize_320=False, eye_oneside_noise = 0,
        face_noise = True,
        flip_mode = 'off',
        img_channel_chw = True,
        augmenter_base=None,
        augmenter_dir=consts.AUG_DIR_MATERIALS, augmenter_cfg_path=None,
        augmenter_options:dict={},
        enable_w_wrt_subject_id = False,
        mixup=False, mixup_halfandhalf=False, mixup_thresh=0.0,
        skip_val=False,
        bin_args={}, # REQUIRED: pitch_nbins, yaw_nbins, pitch_bias, [bin_size]
        dataset_base=None,
        child_of_hybrid=False,
        **kwargs):
        '''
        Same loading / augment / mixup behaviour as the original Dataset_GC3d.
        Removed (vs. the original): bin_sampler, target_N, self.sampler,
            same_person_sample_num (SSL is in the hybrid now), little_mode,
            no_sample/population (replaced by n_per_person below).

        bin_args defines the hybrid grid and is REQUIRED.
        inputs/labels must be a superset of what the hybrid reads from this sub.

        GC3D-exclusive attributes
        n_per_person: per-subject target count. 0 (default) = no balancing
            (every valid sample once). >0 = balance each subject up to n_per_person
            by repetition (+ a random tail), as the old GC3D_Sampler did.
        subjects: if not None (a list of string IDs), only take these subjects.
        subjects_excluded: samples of these subjects are discarded.
        child_of_hybrid: offsets subject_cid (+1000) so ids don't collide with the
            other datasets when used inside a hybrid.

        Supported inputs & labels fields:
        face, eye_L, eye_R, gaze, headpose, bin, fn, subject_cid, gaze_L, gaze_R,
        facemap, facemap_eyelid, eyemap_L/R, irismap_L/R, wE_L/R, wI_L/R
        '''
        self.root = root
        self.split = split
        self.debug = debug
        self.dirs = _DIRS

        if path_gt is None:
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
        self.rgb_no_norm = rgb_no_normalization
        self.return_tensor = return_tensor
        self.child_of_hybrid = child_of_hybrid
        self.n_per_person = n_per_person

        # resolve subject roster filter
        if subjects is not None:
            self.subjects = subjects
        elif self.split=='train' and skip_val:
            print("[GC3D]Warning: skip_val = True. Will use val data for training!")
            self.subjects = None
        else:
            self.gt_sample_list = np.load(path_npy_train_sample, allow_pickle=True).item() if self.split == 'train' else np.load(path_npy_val_sample, allow_pickle=True).item()
            self.subjects = self.gt_sample_list['id'] # list of subjects (5-digit str)
        self.subjects_ex = subjects_excluded

        # bins + valid_ids on the hybrid grid (replaces the old bin/sampler block).
        # _filter_dataset_specific() builds the roster + per-subject balancing.
        self._setup_bins_and_valid_ids(bin_args, self.gt_data['gaze'], self.gt_data.get('headpose'))

    # ===================================================================
    # Dataset-specific filtering: original _filter_ids + per-subject balancing
    # (moved here from GC3D_Sampler).
    # ===================================================================
    def _filter_dataset_specific(self):
        invalid_ids = np.load(os.path.join(self.root, PATH_INVALID_ID), allow_pickle=True)
        invalid = set(int(x) for x in invalid_ids)

        roster = {}
        for i in self.id_list_o:
            if i in invalid:
                continue
            fn = self.gt_data['fn'][i]
            subject = fn[:5]
            if self.subjects is not None and subject not in self.subjects: continue
            if self.subjects_ex is not None and subject in self.subjects_ex: continue
            valid_flag = True
            for name in consts.IMG_TYPES:
                if name not in self.all: continue
                ext = '.png' if name in consts.PNG_TYPES else ".jpg"
                path = os.path.join(self.root, self.dirs[name], fn+ext)
                if not os.path.isfile(path):
                    valid_flag = False
                    break
            if valid_flag:
                roster.setdefault(subject, []).append(i)
        self.D_roster = roster

        # per-subject balancing (was GC3D_Sampler). n_per_person==0 -> no balancing.
        if self.n_per_person and self.n_per_person > 0:
            balanced = []
            for sid, ids in roster.items():
                n_rep = self.n_per_person // len(ids)
                n_tail = self.n_per_person % len(ids)
                for _ in range(n_rep):
                    balanced.extend(ids)
                balanced.extend(random.sample(ids, n_tail))
            return balanced
        else:
            return [i for ids in roster.values() for i in ids]

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

    def _get(self, gt_idx, return_dict=False, force_flip=None):
        '''Core of the original __getitem__. `gt_idx` is a gt index.

        force_flip: None -> decide flip from self.flip_mode (original behaviour).
                    True/False -> pin flip for this call (used by the hybrid's
                    n_views flip-views path so flip is deterministic per view).'''
        index = int(gt_idx)
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
        if force_flip is not None:   # hybrid views: pin flip deterministically per view
            flip = bool(force_flip)

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
            elif name == 'dataset_id':
                values[name] = self.dataset_id
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
                                                      flip=flip,
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

    def update_aug_states(self):
        # for Joe's updates. Call this whenever an self.aug.* is changed externally.
        self.augment_glasses = self.aug.args.glasses.switch and self.split=='train' and PATH_SUBJECT_LMK is not None
        self.face_apply_mask = self.aug.args.add_mask.switch and self.dirs['facemask'] is not None and self.split=='train'
        self.face_replace_green = self.aug.args.replace_green.switch and self.dirs['greenmask'] is not None and self.split=='train'


if __name__ == '__main__':
    BA = {'pitch_nbins': 15, 'yaw_nbins': 17, 'pitch_bias': -12, 'bin_size': 4}
    d = Dataset_GC3d_V2(bin_args=BA, n_per_person=100)
    print(len(d.valid_ids))
