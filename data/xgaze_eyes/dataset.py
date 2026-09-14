'''Dataset classes for TF1.'''
import os
import numpy as np
import random
import cv2
from data.xgaze_eyes.sampler import XGaze_2D_Sampler
from tools.utils import image_preprocess

class Dataset_XGaze_singleeye:
    def __init__(self, root='/data/xgaze/448/', split='train', debug=False, flip=False, augment=True, side=None, 
        output_bin=False, output_cam=False, **kwargs):

        self.root = root
        path_npy = os.path.join(root, "single_eye_{}_p17y17.npy".format(split)) # split = "train" or "val"
        self.gt_data = np.load(path_npy, allow_pickle=True).item()

        self.dir_aug = os.path.join(root, "eye_{}_aug_0415")
        self.dir_o = os.path.join(root, "eye_{}_aug_0415")
        self.aug_num = 50

        self.mean = [0.73044, 0.61, 0.53237]
        self.std = [0.21207, 0.23261, 0.24406]
        
        self.root = root
        self.split = split
        self.num_data = len(self.gt_data['id'])
        
        self.augment = False
        if augment and self.split=='train':
            self.augment=True
            
        self.ids_shuffled = [i for i in range(self.num_data)]
        self.debug = debug
        self.flip = flip

        self.bin = output_bin
        self.cam = output_cam

        self.nbins_pitch = self.gt_data['pitch_nbins']
        self.nbins_yaw = self.gt_data['yaw_nbins']

        self.id_list = []
        self.side = side

        # self.n_valid_cam = 6 if exclude_cam_13 else 8 # deprecated
        self.sampler = XGaze_2D_Sampler(self.gt_data, None, side, split)
        self.id_list = self.sampler.id_list

        self.limit = np.pi * 34/180 # clip all gaze labels => [-limit, +limit]

    def shuffle(self):
        self.id_list = self.sampler.sample()
        print("[Dataset]Shuffled.")

    def __len__(self):
        if self.debug:
            if self.split=='train':
                return 25600
            else: return 12800
        return len(self.id_list)

    def __getitem__(self, i):  # 150 x 90
        # image, gt (2)
        # image, gt (2), gt_bin (2) | bin=True
        # image, gt (2), gt_bin (2), cam_id | cam=True (bin must bin True)
        index = self.id_list[i]
        c = self.gt_data['side'][index]
        id_eq = self.gt_data['id'][index]
        cam_id = self.gt_data['cam_cid'][index]
        
        if self.augment:
            j = random.randrange(self.aug_num)
            image_path = os.path.join(self.root, self.dir_aug.format(c), str(id_eq), str(j)+'.jpg')
        else:
            image_path = os.path.join(self.root, self.dir_o.format(c), str(id_eq)+'.jpg')

        if not os.path.isfile(image_path):
            raise Exception("[Dataset]Image not found: %s" % image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = [image]

        gt = self.gt_data['gaze'][index]
        # head_pose = self.gt_data['headpose'][index]

        if self.flip and c=='R':
            image = cv2.flip(image, 1)
            gt[1] = (-1.)*gt[1]

        if self.limit:
            gt = np.clip(gt, -self.limit, self.limit) # 17-bins

        result.append(gt)

        if self.bin:
            pitch_bin_id = self.gt_data['pitch_cid'][index]
            yaw_bin_id = self.gt_data['yaw_cid'][index]
            result.append([pitch_bin_id, yaw_bin_id])
            
        if self.cam:
            result.append(cam_id)

        return result

class Dataset_XGaze_doubleeye:
    def __init__(self, root='/data/xgaze/448/', split='train', debug=False, augment=True, 
        output_bin=False, output_cam=False, oneside_noise=0, preprocess=True, **kwargs):
        # oneside_noise: attach noise to L or R, to assist weighting branch training
        # 0: switched off
        # (0,1]: probability to attach noise

        self.root = root
        path_npy = os.path.join(root, "double_eye_{}_p17y17.npy".format(split)) # split = "train" or "val"
        self.gt_data = np.load(path_npy, allow_pickle=True).item()

        self.dir_aug = os.path.join(root, "eye_{}_aug_0415")
        self.dir_o = os.path.join(root, "eye_{}_aug_0415")
        self.aug_num = 50
        
        self.root = root
        self.split = split
        self.num_data = len(self.gt_data['id'])

        self.mean = [0.73044, 0.61, 0.53237]
        self.std = [0.21207, 0.23261, 0.24406]
        
        self.augment = False
        if augment and self.split=='train':
            self.augment=True
            
        self.ids_shuffled = [i for i in range(self.num_data)]
        self.debug = debug

        self.bin = output_bin
        self.cam = output_cam
        self.preprocess = preprocess
        self.oneside_noise = oneside_noise

        self.nbins_pitch = self.gt_data['pitch_nbins']
        self.nbins_yaw = self.gt_data['yaw_nbins']

        self.id_list = []
        self.n_valid_cam = None

        self.sampler = XGaze_2D_Sampler(self.gt_data, self.n_valid_cam, None, split)
        self.id_list = self.sampler.id_list

        self.limit = np.pi * 34/180 # clip all gaze labels => [-limit, +limit]

    def shuffle(self):
        self.id_list = self.sampler.sample()
        print("[Dataset]Shuffled.")

    def __len__(self):
        if self.debug:
            if self.split=='train':
                return 25600
            else: return 12800
        return len(self.id_list)

    def __getitem__(self, i):  # 150 x 90
        # (imageL, imageR), gt (2)
        # (imageL, imageR), gt (2), gt_bin (2) | bin=True
        # (imageL, imageR), gt (2), gt_bin (2), cam_id | cam=True (bin must bin True)
        index = self.id_list[i]
        id_eq = self.gt_data['id'][index]
        cam_id = self.gt_data['cam_cid'][index]
        
        images = []

        c0 = None
        if self.oneside_noise>0 and random.random()<self.oneside_noise:
            c0 = 'L' if random.random()<0.5 else 'R'

        for c in ('L','R'):
            if self.augment:
                j = random.randrange(self.aug_num)
                image_path = os.path.join(self.root, self.dir_aug.format(c), str(id_eq), str(j)+'.jpg')
            else:
                image_path = os.path.join(self.root, self.dir_o.format(c), str(id_eq)+'.jpg')

            if not os.path.isfile(image_path):
                raise Exception("[Dataset]Image not found: %s" % image_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if c0==c:
                image = image_preprocess(image, self.mean, self.std, jitter=False, noise_std = 0.25)
            elif self.preprocess:
                image = image_preprocess(image, self.mean, self.std, jitter=False)

            images.append(image)
        
        result = [images]

        gt = self.gt_data['gaze'][index]
        # head_pose = self.gt_data['headpose'][index]
        
        if self.limit:
            gt = np.clip(gt, -self.limit, self.limit) # 17-bins

        result.append(gt)

        if self.bin:
            pitch_bin_id = self.gt_data['pitch_cid'][index]
            yaw_bin_id = self.gt_data['yaw_cid'][index]
            result.append([pitch_bin_id, yaw_bin_id])
            
        if self.cam:
            result.append(cam_id)

        return result