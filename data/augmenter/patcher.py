import os
import cv2
import random
import numpy as np

from data.consts import *

class Patcher:
    '''
    a combination of old Green_Background_Replacer & Glasses_Simulator.
    Anyways, a Patcher makes use of an image library and paste (part of) images to a dataset sample
    '''
    def __init__(self, dir_imgs=None, size=224):
        # <dir_imgs>/list.npy contains the image list in <dir_imgs>. If it doesn't exist, I will generate one.
        # size <= 256.
        self.root = dir_imgs
        self.path_list = os.path.join(self.root, "image_list.npy")
        if not os.path.isfile(self.path_list):
            self.generate_meta()
        else:
            self.fnlist = np.load(self.path_list, allow_pickle=True)
        self.n = len(self.fnlist)

        self.size = size

    def generate_meta(self):
        import glob
        self.fnlist = glob.glob(self.root + "**/*.jpg")
        np.save(self.path_list, self.fnlist)

    def __len__(self):
        return len(self.fnlist)

    def grab_one_background(self):
        '''
        Grab one image from the library and crop it to target size
        '''
        w = 0; h = 0
        while w<self.size or h<self.size: 
            i = random.randrange(self.n)
            path = os.path.join(self.root, self.fnlist[i])
            rgb = cv2.imread(path)
            if rgb is None: continue
            rgb = rgb[:,:,::-1]
            # random crop
            h, w, _ = rgb.shape
 
        x0 = random.randrange(w - self.size + 1)
        y0 = random.randrange(h - self.size + 1)
        return rgb[y0:y0+self.size, x0:x0+self.size, :]

    def replace_green_background(self, rgb, green_mask, solid_color_prob=0.05):
        # both rgb & green_mask are float (0-1).
        if random.random()<solid_color_prob:
            color = (random.random(), random.random(), random.random())
            bg = np.ones((self.size, self.size, 3)) * color
        else:
            bg = self.grab_one_background()
            bg = bg.astype(np.float32)/255
        # RGB & Background needed to be consistent --> [0,255]
        rgb = (1 - green_mask) * rgb + green_mask * bg * 255
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb
    '''
    Below are Glasses_Simulator
    '''
    @staticmethod
    def pick(sl:slice, l=7):
        ids = list(range(sl.start, sl.stop))
        perm = np.random.permutation(l)
        num = np.random.randint(3, 5)
        picked = sorted([perm[i] for i in range(num)])
        return [ids[picked[i]] for i in range(num)]
    
    def simulate_glasses_reflection(self, rgb, mask, glasses_opacity, frame_select='none', solid_color_prob=0.05):

        gls_left = mask[:,:,1][:,:,None] // 255
        gls_right = mask[:,:,2][:,:,None] // 255
        if frame_select == 'both':
            glasses_opacity_left, glasses_opacity_right = glasses_opacity[0], glasses_opacity[1]
        elif frame_select == 'right':
            glasses_opacity_left, glasses_opacity_right = 0, glasses_opacity[1]
        elif frame_select == 'left':
            glasses_opacity_left, glasses_opacity_right = glasses_opacity[0], 0
        elif frame_select == 'none':
            glasses_opacity_left, glasses_opacity_right = 0, 0
        elif frame_select == 'random':
            glasses_opacity_left, glasses_opacity_right = random.randint(0, 1)*glasses_opacity[0], random.randint(0, 1)*glasses_opacity[1]
        else:
            raise("frame Select must be valid [both, left, right, none]")
        
        if random.random() < solid_color_prob:
            glass_mask_left = mask[:,:, 1][:,:,None] * np.random.rand(3)
            glass_mask_right = mask[:,:, 2][:,:,None] * np.random.rand(3)
            bg = glass_mask_left + glass_mask_right
        else:
            bg = self.grab_one_background()
        
        # gls_mask = cv2.resize(gls_mask, (rgb.shape[0], rgb.shape[1]))[:,:, None]
        # x = bg * gls_mask
      
        result = rgb * (1 - glasses_opacity_left * gls_left - glasses_opacity_right * gls_right) + glasses_opacity_left * bg * gls_left + glasses_opacity_right * bg * gls_right
        result = result.astype(np.uint8)
        return result

    def simulate_glasses(self, rgb_normalized, landmarks34, solid_color_prob=0.05):
        if random.random()<solid_color_prob:
            color = (random.random(), random.random(), random.random())
            bg = np.ones((self.size, self.size, 3)) * color
        else:
            bg = self.grab_one_background()
            bg = bg.astype(np.float32)/255

        from tools.utils import mix2

        # generate polygons
        mask = np.zeros((self.size, self.size, 3))
        
        # left
        centroid_l = landmarks34[GS_ID_CENTROID_L]

        ids_lt = self.pick(GS_SL_LT)
        ids_lb = self.pick(GS_SL_LB)
        ids_poly_left = ids_lt + ids_lb
        ld_poly_left = landmarks34[ids_poly_left, :]

        # random interpolation
        n_left = len(ids_poly_left)
        ld_cen_left = np.tile(centroid_l, (n_left, 1))
        coeff = np.random.rand(n_left, 1) * 0.5 + 0.5
        poly_left = ld_poly_left * coeff + ld_cen_left * (-coeff + 1)

        hull = cv2.convexHull(poly_left.astype(np.int32)).squeeze()
        cv2.fillPoly(mask, [hull], [1,1,1])

        # right
        centroid_r = landmarks34[GS_ID_CENTROID_R]

        ids_rt = self.pick(GS_SL_RT)
        ids_rb = self.pick(GS_SL_RB)
        ids_poly_right = ids_rt + ids_rb
        ld_poly_right = landmarks34[ids_poly_right, :]

        # random interpolation
        n_right = len(ids_poly_right)
        ld_cen_right = np.tile(centroid_r, (n_right, 1))
        coeff = np.random.rand(n_right, 1) * 0.5 + 0.5
        poly_right = ld_poly_right * coeff + ld_cen_right * (-coeff + 1)

        hull = cv2.convexHull(poly_right.astype(np.int32)).squeeze()
        cv2.fillPoly(mask, [hull], [1,1,1])

        alpha = random.random() * 0.3 + 0.5
        out = mix2(rgb_normalized, bg, mask, alpha)[:,:,::-1]

        return out
