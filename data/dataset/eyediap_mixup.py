'''
Eyediap dataset class.
Apr '23: updated by Julia
    * reworked 3D gaze labels
    * new arg `session`: "CS/DS" support

July 10 '23: updated by Nina
    * added mixup feature for eyediap dataset

July 18 '23: updated by Amirul
    * added facemap_v3

Aug 09 '23 updated by Nina
    * allow session = ['cs', 'ds']
'''
import os
import cv2
import torch
import random
import numpy as np

from data.augmenter.augmenter import initialize_augmenter
import data.consts as consts

from torch.utils.data import Dataset as D

from data.mixup import MixupProcessor

ROOT = '/data/eyediap/'
FOLDS = [0, 1, 2, 3]
# if using Beihang dataset use or pass from .yaml:
PATH_GT = ['gt_eyediap_cs_with_conf_v3_bh.npy', 'gt_eyediap_ds_with_conf_v3_bh.npy']
_DIRS = {
    'face': ['normalized_face_224_cs_bh', 'normalized_face_224_ds_bh'],
    'facemap_v3': ['seg_cs_bh', 'seg_ds_bh']
}

# PATH_GT = ['gt_eyediap_cs_with_conf_v3.npy', 'gt_eyediap_ds_with_conf_v3.npy']

# _DIRS = {
#     'face': ['normalized_face_224_cs', 'normalized_face_224_ds']
# }

_DIRS_MIX = 'mixup_matrices'

_MIXUP_FILES = {
    'cs': ['top_pair_save_no_duplicate_cs.npy', 'second_pair_save_no_duplicate_cs.npy'],
    'ds': ['top_pair_save_no_duplicate_ds.npy', 'second_pair_save_no_duplicate_ds.npy'],
    'both': ['top_pair_save_no_duplicate_cs_and_ds.npy', "second_pair_save_no_duplicate_cs_and_ds.npy"]
}

class Dataset_Eyediap(D):
    def __init__(self, root=ROOT, path_gt=PATH_GT, dirs=_DIRS,
                 split='train', debug=False, augment=True,
                 inputs=['face'], labels=['gaze'], session=['cs', 'ds'],
                 val_subjects: list = [],
                 rgb_no_normalization=False, return_tensor=False,
                 flip_mode='off',
                 mixup = False, mixup_halfandhalf = False, mixup_num_samples_cap = None,
                 img_channel_chw=True,
                 augmenter_dir=consts.AUG_DIR_MATERIALS, augmenter_cfg_path=None,
                 augmenter_options: dict = {},
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
        val_subjects: val_folds, the fold(s) (#0-3) for validation.
            so, if split==train, the Dataset uses the complement folds.
            If ==[], always apply all folds.
            used val_subjects for consistency with other datasets
        ###LZH: session = 'cs' or 'ds', not a LIST.###
        session: continuous screen ('cs') ['cs'] or discrete screen ('ds') ['ds'] sessions or both ['cs', 'ds']

        mixup:
            * if True, generate online mixup images. For training only.
            * mixup_halfandhalf:
                * False (default): mixup ratio = rand()
                * True: mixup ratio = 0.5 always
            * mixup_num_samples_cap: if None, manually impose a cap to len(self). (debug sets it to 256)
                * For training only.
        '''
        # filter ground truth paths based on session
        gt = []
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

        self.dirs = dirs

        self.inputs = inputs
        self.labels = labels
        self.all = inputs + labels

        self.session = session

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

        self.mixup = mixup
        self.mixup_halfandhalf = mixup_halfandhalf

        # TODO: change this part
        mix_mat1 = []
        mix_mat2 = []

        if len(session) == 1:
            self.cs_and_ds = False
            for s in session:
                mix_mat1 = np.load(os.path.join(ROOT, _DIRS_MIX, _MIXUP_FILES[s][0])).astype(int)
                mix_mat2 = np.load(os.path.join(ROOT, _DIRS_MIX, _MIXUP_FILES[s][1])).astype(int)
        else:
            self.cs_and_ds = True
            mix_mat1 = np.load(os.path.join(ROOT, _DIRS_MIX, _MIXUP_FILES['both'][0])).astype(int)
            mix_mat2 = np.load(os.path.join(ROOT, _DIRS_MIX, _MIXUP_FILES['both'][1])).astype(int)

        self.mixup_matrix = np.concatenate((mix_mat1, mix_mat2), axis=1)

        if self.debug:
            print("[Debug] Total number of none -1 mixes: ", (self.mixup_matrix >= 0).sum())
            print("[Debug] Number of images: ", len(self.gt_data['fn']))
        # edit loading mixup matrix end here

        self.num_data_o = len(self.gt_data['fn'])

        is_training = split in ['train', 'train_cv', 'val_cv']

        if is_training and self.mixup:
            self.id_list_o = list(range(self.num_data_o * 32))
        else:
            self.id_list_o = list(range(self.num_data_o))

        self.flip_mode = flip_mode if self.split == 'train' else 'off'

        self.folds = val_subjects
        self.chw = img_channel_chw
        self.rgb_no_norm = rgb_no_normalization
        self.return_tensor = return_tensor

        # self.valid_ids = self._filter_ids()
        # self.id_list = list(range(self.num_data))
        # self.id_list = self._filter_ids(is_training)
        if self.debug:
            print("[debug]: entering _filter_ids")

        self.id_list = self._filter_ids(is_training)
        self.num_data = len(self.id_list)
        print("[Dataset]Valid ID list generated. # =", self.num_data)

        # for debug
        self.num_data_cap = mixup_num_samples_cap

        self.augment = augment and self.split == 'train'
        self.augment_args = augmenter_options
        self.aug = initialize_augmenter(self.split, augmenter_dir, augmenter_cfg_path)

    def __len__(self):
        if self.debug:
            return 512
            # return self.num_data
        # TODO: copied from mpiiface
        if self.split=='train_cv' and self.num_data_cap is not None:
            return self.num_data_cap

        return self.num_data

    def _get_mixup_id(self, idx, is_training):

        subject_cids = []
        image_cids = []
        fold_id = []

        # in case of validation or mixup is false
        if is_training is False or self.mixup is False:

            img_id = int(idx)
            sbjt_cid = self.gt_data['subject_cid'][img_id]
            fold = self.gt_data['fold_id'][img_id]

            subject_cids.append(sbjt_cid)
            image_cids.append(img_id)
            fold_id.append(fold)

            subject_cids.append(None)
            image_cids.append(None)
            fold_id.append(None)

            return subject_cids, image_cids, fold_id

        # img id in range 6487 if session = ['cs']
        # img id in range 8337 if session = ['ds']
        # img id in range 14824 if session = ['cs, 'ds']

        # information of first image
        img_id = int(np.floor(idx/32))
        sbjt_cid = self.gt_data['subject_cid'][img_id]
        fold = self.gt_data['fold_id'][img_id]

        subject_cids.append(sbjt_cid)
        image_cids.append(img_id)
        fold_id.append(fold)

        # subject_cid of the pairing image
        match_img_sid = int(idx % 32)

        # case when mixup is with itself
        if self.mixup_matrix[img_id][match_img_sid] == -1 and (sbjt_cid == match_img_sid):
            subject_cids.append(None)
            image_cids.append(None)
            fold_id.append(None)

        # case when no successful mixup with a different subject_cid
        elif self.mixup_matrix[img_id][match_img_sid] == -1 and sbjt_cid != match_img_sid:
            return [None, None], [None, None], [None, None]

        # case of a successful mix
        else:
            mix_img_id = self.mixup_matrix[img_id][match_img_sid]
            subject_cids.append(self.gt_data['subject_cid'][mix_img_id])
            image_cids.append(mix_img_id)
            fold_id.append(self.gt_data['fold_id'][mix_img_id])

        # subject cids in 0-15, image cids in 0-(6487+8337-1), fold id in 0-3
        return subject_cids, image_cids, fold_id

    def _filter_ids(self, is_training):
        valid_ids = []
        ext = '.png'

        self.mixup_id1 = []
        self.mixup_id2 = []

        if len(self.folds) > 0:
            # exclude specified folds when in training
            if self.split == 'train':

                valid_folds = [f for f in FOLDS if f not in self.folds]
                if self.debug:
                    print("[debug]: valid folds: ", valid_folds)
            # if validation
            else:
                valid_folds = self.folds
        # not specified
        else:
            valid_folds = FOLDS

        for i in self.id_list_o:
            subject_cids, image_cids, fold_ids = self._get_mixup_id(i, is_training)
            self.mixup_id1.append(image_cids[0])
            self.mixup_id2.append(image_cids[1])
            valid_flag = True
            # case when no mix
            if subject_cids[0] is None:
                continue
            # case no mix
            if subject_cids[1] is None:
                fid = fold_ids[0]
                if fid not in valid_folds:
                    continue
                for name in consts.IMG_TYPES:
                    if name not in self.all: continue
                    # path = []
                    # in this case is cs and ds together
                    if self.cs_and_ds:
                        # first 6487 ids correspond to cs images
                        if image_cids[0] <= 6486:
                            path = os.path.join(self.root, self.dirs[name][0], self.gt_data['fn'][image_cids[0]]+ext)
                            # print("[debug]: in filter no mix case cs")
                        # case in ds
                        else:
                            path = os.path.join(self.root, self.dirs[name][1], self.gt_data['fn'][image_cids[0]]+ext)
                    # in case cs ds either or, then length of path will always be 1
                    else:
                        path = []
                        # there should only be one path (cs or ds)
                        for temp_dir in self.dirs[name]:
                            if any(s in temp_dir for s in self.session):
                                path = (os.path.join(self.root, temp_dir, self.gt_data['fn'][image_cids[0]] + ext))

                    if not os.path.isfile(path):
                        valid_flag = False
                        break
                if valid_flag:
                    valid_ids.append(i)
                # case for mixup
            else:
                if fold_ids[0] not in valid_folds or fold_ids[1] not in valid_folds:
                    continue
                for name in consts.IMG_TYPES:
                    if name not in self.all: continue

                    # case cs and ds together
                    if self.cs_and_ds:
                        if image_cids[0] <= 6486:
                            path_img1 = os.path.join(self.root, self.dirs[name][0], self.gt_data['fn'][image_cids[0]]+ext)
                            # print("[debug]: in filter no mix case cs")
                        # case in ds
                        else:
                            path_img1 = os.path.join(self.root, self.dirs[name][1], self.gt_data['fn'][image_cids[0]]+ext)

                        if image_cids[1] <= 6486:
                            path_img2 = os.path.join(self.root, self.dirs[name][0], self.gt_data['fn'][image_cids[1]]+ext)
                            # print("[debug]: in filter no mix case cs")
                        # case in ds
                        else:
                            path_img2 = os.path.join(self.root, self.dirs[name][1], self.gt_data['fn'][image_cids[1]]+ext)
                    # case cs or ds only
                    else:
                        path_img1 = []
                        path_img2 = []
                        for temp_dir in self.dirs[name]:
                            if any(s in temp_dir for s in self.session):
                                path_img1 = os.path.join(self.root, temp_dir, self.gt_data['fn'][image_cids[0]] + ext)
                                path_img2 = os.path.join(self.root, temp_dir, self.gt_data['fn'][image_cids[1]] + ext)

                    if not os.path.isfile(path_img1) or not os.path.isfile(path_img2):
                        # print("[debug]: no file exits in path")
                        valid_flag = False
                        break
                if valid_flag:
                    valid_ids.append(i)

        self.num_data = len(valid_ids)
        print("[Dataset]Valid ID list generated. # =", self.num_data)
        if self.debug:
            print("[Debug]: last element of the valid id list: ", valid_ids[-1])
        return valid_ids

    def _load_rgb(self, type: str, i: int, alpha = 0.5, **kwargs):
        '''
        type in ("face")
        We have face images only at the moment.
        '''
        ext = '.png'
        # no mix
        if self.mixup_id2[i] is None:
            if self.debug:
                print("[Debug][Load RGB]: Not Mixup")

            # if cs and ds together
            if self.cs_and_ds:
                if self.mixup_id1[i] <= 6486:
                    path = os.path.join(self.root, self.dirs[type][0], self.gt_data['fn'][self.mixup_id1[i]] + ext)
                # case in ds
                else:
                    path = os.path.join(self.root, self.dirs[type][1], self.gt_data['fn'][self.mixup_id1[i]] + ext)
            # cs or ds only
            else:
                path = []
                for temp_dir in self.dirs[type]:
                    if any(s in temp_dir for s in self.session):
                        path = (os.path.join(self.root, temp_dir, self.gt_data['fn'][self.mixup_id1[i]] + ext))

            if not os.path.isfile(path):
                print("[Dataset]Warning: image {} not exists. Return None.".format(path))
                return None
            if type not in consts.RGB_TYPES and self.mixup:  # needed to match the shape of the tensors in the batch
                segmap = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                rgb = np.concatenate((segmap, segmap), axis=-1)
            else:
                rgb = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # case mixup
        else:
            if self.debug:
                print("[Debug][Load RGB]: Mixup")
            # cs and ds together
            if self.cs_and_ds:
                if self.mixup_id1[i] <= 6486:
                    path_img1 = os.path.join(self.root, self.dirs[type][0], self.gt_data['fn'][self.mixup_id1[i]] + ext)
                # print("[debug]: in filter no mix case cs")
                # case in ds
                else:
                    path_img1 = os.path.join(self.root, self.dirs[type][1], self.gt_data['fn'][self.mixup_id1[i]] + ext)

                if self.mixup_id2[i] <= 6486:
                    path_img2 = os.path.join(self.root, self.dirs[type][0], self.gt_data['fn'][self.mixup_id2[i]] + ext)
                # print("[debug]: in filter no mix case cs")
                # case in ds
                else:
                    path_img2 = os.path.join(self.root, self.dirs[type][1], self.gt_data['fn'][self.mixup_id2[i]] + ext)
            # cs or ds only
            else:
                path_img1 = []
                path_img2 = []
                for temp_dir in self.dirs[type]:
                    if any(s in temp_dir for s in self.session):
                        path_img1 = os.path.join(self.root, temp_dir, self.gt_data['fn'][self.mixup_id1[i]] + ext)
                        path_img2 = os.path.join(self.root, temp_dir, self.gt_data['fn'][self.mixup_id2[i]] + ext)

            # when two images are face
            if type in consts.RGB_TYPES:
                rgb = MixupProcessor.mix_images(path_img1, path_img2, alpha)
            else:
                segmap_1 = cv2.imread(path_img1, cv2.IMREAD_UNCHANGED)
                segmap_2 = cv2.imread(path_img1, cv2.IMREAD_UNCHANGED)
                rgb = np.concatenate((segmap_1, segmap_2), axis=-1)

        if type in consts.RGB_TYPES:
            rgb = rgb[:, :, ::-1]

        return rgb

    def __getitem__(self, i):
        index = self.id_list[i]
        if self.debug:
            print("[debug]: index: ", index)
            print("[Debug]: index image 1: ", self.mixup_id1[index])
            print("[Debug]: info image 1: ", self.gt_data['fn'][self.mixup_id1[index]],
                self.gt_data['fold_id'][self.mixup_id1[index]], self.gt_data['subject_cid'][self.mixup_id1[index]])
            print("[Debug]: index image 2: ", self.mixup_id2[index])
            if self.mixup_id2[index] is not None:
                print("[Debug]: info image 2: ", self.gt_data['fn'][self.mixup_id2[index]],
                    self.gt_data['fold_id'][self.mixup_id2[index]], self.gt_data['subject_cid'][self.mixup_id2[index]])
        values = {}

        if self.mixup_halfandhalf:
            alpha = 0.5
        else:
            alpha = random.random()

        is_mixup = self.mixup_id2[index] is not None

        # global random values
        flip = False
        if self.split == 'train':
            if self.flip_mode == 'on' or (self.flip_mode == 'random' and random.random() < 0.5):
                flip = True

        img_dict = {}

        for name in self.all:
            if name in consts.ANGULAR_TYPES:
                if not is_mixup:
                    val = self.gt_data[name][self.mixup_id1[index]]
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
                    val = self.gt_data[name][self.mixup_id1[index]]
                    values[name] = val
                else:
                    values[name] = [self.gt_data[name][self.mixup_id1[index]],
                                    self.gt_data[name][self.mixup_id2[index]]]

            elif name in consts.CONST_TYPES or name == 'seg_w': #  mixup exclusive grab the confidence scores and store in a list
                if not is_mixup:
                    # print(index, len(self.gt_data['wE_L']), self.mixup_id1[index], self.mixup_id2[index])
                    wE_L = self.gt_data['wE_L'][self.mixup_id1[index]]
                    wI_L = self.gt_data['wI_L'][self.mixup_id1[index]]
                    wE_R = self.gt_data['wE_R'][self.mixup_id1[index]]
                    wI_R = self.gt_data['wI_R'][self.mixup_id1[index]]
                    val = np.array([wE_L, wI_L, wE_R, wI_R])

                    if self.mixup: # needed to match the shape of the tensors in the batch
                        values[name] = np.concatenate((val, val), axis=-1)
                    else:
                        values[name] = val
                else:
                    img1_num = self.mixup_id1[index]
                    img2_num = self.mixup_id2[index]

                    wE_L_1 = self.gt_data['wE_L'][img1_num]
                    wI_L_1 = self.gt_data['wI_L'][img1_num]
                    wE_R_1 = self.gt_data['wE_R'][img1_num]
                    wI_R_1 = self.gt_data['wI_R'][img1_num]
                    values_num1 = np.array([wE_L_1, wI_L_1, wE_R_1, wI_R_1])

                    wE_L_2 = self.gt_data['wE_L'][img2_num]
                    wI_L_2 = self.gt_data['wI_L'][img2_num]
                    wE_R_2 = self.gt_data['wE_R'][img2_num]
                    wI_R_2 = self.gt_data['wI_R'][img2_num]
                    values_num2 = np.array([wE_L_2, wI_L_2, wE_R_2, wI_R_2])

                    values[name] = np.concatenate((values_num1, values_num2), axis=-1) # concate the list of confidence scores of two samples

            elif name == consts.ALPHA:# mixup-exclusive
                values[name] = alpha
            else: #'face'
                img_dict[name] = self._load_rgb(name, index, alpha)

        # batch-augment
        if not self.rgb_no_norm and len(img_dict) > 0:
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
