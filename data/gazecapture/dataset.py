import os
import yaml
import glob
import h5py
import numpy as np
import random
import cv2
import csv
import utils_lzh


def image_preprocess(rgb, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), jitter=False, black_border=False, replace_green=None, noise=False, do_clahe=False):
    h,w,_ = rgb.shape # w=150, h=90
    if rgb.dtype=='uint8':
        rgb = rgb.astype("float32")
        rgb_max = np.amax(rgb)
        # print(rgb_max)
        rgb /= rgb_max # then it's [0,1]

#    t1 = time.time()
    if replace_green and os.path.isdir(replace_green):
        import glob
        n_bg_imgs = 47
        id_bg = np.random.randint(n_bg_imgs+1)
        if id_bg<n_bg_imgs:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            l_green = np.array([70., 0.5, 0.])
            u_green = np.array([170., 1., 1.])
            mask = cv2.inRange(hsv, l_green, u_green)
            # mask = guidedFilter(rgb, mask, 5, 1e-6)
            if np.amax(mask)>128:
                id_r = np.random.randint(100)
                #print(bg_imgs[id_bg])
                bg = cv2.imread(os.path.join(DIR_BG, str(id_bg), str(id_r)+'.jpg'))
                bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                bg = bg.astype(float)/255

                h_bg, w_bg, _ = bg.shape
                x1 = np.random.randint(w_bg-w)
                y1 = np.random.randint(h_bg-h)
                # s = np.random.randint(max(w, h), min(w_bg-x1, h_bg-y1))
                bg = bg[y1:y1+h, x1:x1+w, :]
                # bg = cv2.resize(bg, (w, h))

                # scale = s_rgb/min(bg.shape[:2])
                # h, w, _ = bg.shape
                # bg = cv2.resize(bg, (int(w*scale), int(h*scale)))
                # offset = np.random.randint(max(bg.shape)-s_rgb+1)
                # if h>w:
                #     bg = bg[offset:offset+s_rgb, :, :]
                # else:
                #     bg = bg[:, offset:offset+s_rgb, :]

                mask = np.stack([mask,mask,mask], axis=2) / 255
                rgb = (-mask+1) * rgb + mask * bg
                # rgb = np.where(np.stack([mask,mask,mask], axis=2), bg, rgb)
#    t2=time.time()
#    print("Background:", t2-t1)

 #   t1=time.time()
    if jitter:
        gamma_i = 1 + (random.random() - 0.5) #[0.5-1.5]
        # gamma_i = 2.5 ** (random.random()*2. - 1.) #[0.5-2]

        # rgb = rgb ** gamma_i
        expo = np.zeros_like(rgb)
        for c in range(3):
            gamma_rgb = 1 + (random.random() - 0.5) / 2  # [0.75-1.25]
            # rgb[..., c] = rgb[..., c] ** gamma_rgb
            expo[:,:,c] = gamma_rgb+gamma_i
        
        rgb = np.power(rgb, expo)
#    t2=time.time()

    if noise and random.random()>0.5:
        noise_std = random.random() * 0.1 + 0.05
        noise_map = np.random.normal(0, noise_std, (h,w,3))
        rgb += noise_map
        rgb = np.clip(rgb, 0., 1.)

#    t1=time.time()
# black_border=False for eyes (skipped)
#     if black_border and random.random()>0.4: # and np.random.rand()>0.5:
#         r_id = int(np.round(np.random.normal(15,7.5)))
#         r_id = np.clip(r_id, 0, 30)
#         shift = int(np.round(np.random.normal(0, 30)))

#         from_top = np.random.rand()>0.5
#         mask = cv2.imread(DIR_ROT31 + "/%02d.png" % r_id).astype(float) / 255
#         if from_top:
#             shift+=224
#         else:
#             shift-=224
#             mask = -mask + 1
#         mask = mask[750-224+shift:750+224+shift, ...]
#         rgb = rgb*mask
#  #   t2=time.time()

  #  t1=time.time()
    if jitter:
        # random scaling & crop
        a = int(np.abs(np.round(np.random.normal(0, 2))))
        b = int(np.abs(np.round(np.random.normal(0, 2))))
        if b==0:
            rgb = rgb[a:, :, :]
        else: 
            rgb = rgb[a:-b, :, :]

        a = int(np.abs(np.round(np.random.normal(0, 3))))
        b = int(np.abs(np.round(np.random.normal(0, 3))))
        if b==0:
            rgb = rgb[:, a:, :]
        else: 
            rgb = rgb[:, a:-b, :]
        rgb = cv2.resize(rgb, (w, h))
#    t2=time.time()

# do histogram equalization
    if do_clahe:
        hls = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2HLS)
        L = (hls[...,1]*255).astype('uint8')

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L_eq = clahe.apply(L)

        # L_eq = cv2.equalizeHist(L)
        hls[...,1] = L_eq.astype(np.float32)/255.
        rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    return rgb


class Dataset_Gazecapture_eye:
    def __init__(self, root='/data/gazecapture_3d/', split='train', debug=False, augment=True, 
        output_bin=False, oneside_noise=0, preprocess=True, **kwargs):
        # oneside_noise: attach noise to L or R, to assist weighting branch training
        # 0: switched off
        # (0,1]: probability to attach noise

        self.root = root
        path_npy = os.path.join(root, "double_eye_{}_p17y17.npy".format(split)) # split = "train" or "val"
        self.gt_data = np.load(path_npy, allow_pickle=True).item() # gt npy data preparing

        self.eye_dir = os.path.join(root, "eye_{}")
        self.aug_num = 50

        self.mean = [0.73044, 0.61, 0.53237] # mean and std need re-calculated
        self.std = [0.21207, 0.23261, 0.24406]
        
        self.root = root
        self.split = split
        self.num_data = len(self.gt_data['id'])
        
        self.augment = False
        if augment and self.split=='train':
            self.augment=True
            
        self.ids_shuffled = [i for i in range(self.num_data)]
        self.debug = debug

        self.bin = output_bin
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
        id_eq = self.gt_data['fn'][index] # npy need to be linked?
        cam_id = None # 'None' in Gazecapture
        
        images = []

        c0 = None
        if self.oneside_noise>0 and random.random()<self.oneside_noise:
            c0 = 'L' if random.random()<0.5 else 'R'

        for c in ('L','R'):
            if self.augment:
                image_path = os.path.join(self.root, self.eye_dir.format(c), str(id_eq)+'.jpg')
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # add xgaze #50 augmentation here
                image = image_preprocess(image, self.mean, self.std, jitter=True, noise=True)
                
            else:
                image_path = os.path.join(self.root, self.eye_dir.format(c), str(id_eq)+'.jpg')
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if not os.path.isfile(image_path):
                raise Exception("[Dataset]Image not found: %s" % image_path)

            if c0==c:
                image = utils_lzh.image_preprocess(image, self.mean, self.std, jitter=False, noise_std = 0.25)
            elif self.preprocess:
                image = utils_lzh.image_preprocess(image, self.mean, self.std, jitter=False)

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