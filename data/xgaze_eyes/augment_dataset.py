'''
Data augmentation function for XGaze.
Can be extended to other datasets later.
'''

import glob
import numpy as np
import os
import random
import cv2
from tqdm import tqdm
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", dest="start", type=int)
parser.add_argument("-e", dest="end", type=int)

def jitter_rgb(rgb):
        gamma_i = 1 + (random.random() - 0.5)  # [0.5-1.5]
        # rgb = rgb ** gamma_i
        for c in range(3):
            gamma_rgb = 1 + (random.random() - 0.5) / 2  # [0.75-1.25]
            # rgb[..., c] = rgb[..., c] ** (gamma_rgb + gamma_i)
            rgb[..., c] = np.power(rgb[..., c], gamma_rgb + gamma_i)
        return rgb

def generate_bg(num=100):
    counter = 0
    for fn in os.listdir(DIR_BG):
        path = os.path.join(DIR_BG, fn)
        if not os.path.isfile(path): continue
        if fn!= 'img_bg_1.jpg':
            counter+=1
            continue
        target_dir = os.path.join(DIR_BG, str(counter))
        if not os.path.isdir(target_dir): os.mkdir(target_dir)
        rgb = cv2.imread(path).astype('float')/255
        rgb = cv2.resize(rgb, (851, 800))
        for i in range(num):
            rgbx = (jitter_rgb(rgb.copy())*255).astype('uint8')
            cv2.imwrite(os.path.join(target_dir, str(i)+'.jpg'), rgbx)
        print("Done:", fn)
        counter+=1
    print("Voila.")

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

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        L_eq = clahe.apply(L)

        # L_eq = cv2.equalizeHist(L)
        hls[...,1] = L_eq.astype(np.float32)/255.
        rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    return rgb
    # return L # returns equalized gray signal
    
def clahe(rgb):
    rgb = rgb.astype(np.float32)
    rgb_max = np.amax(rgb)
    rgb /= rgb_max # then it's [0,1]
    
    hls = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2HLS)
    L = (hls[...,1]*255).astype('uint8')

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    L_eq = clahe.apply(L)

    # L_eq = cv2.equalizeHist(L)
    hls[...,1] = L_eq.astype(np.float32)/255.
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    return rgb
    
    
def manta_style(num=50, start=0, end=-1):
    import shutil
    if end==-1: end=756541
    for i in tqdm(range(start, end)):
        path_source = os.path.join(root_source, str(i)+'.jpg')
        if not os.path.isfile(path_source): continue
        path_target = os.path.join(root_target, str(i))
        # if os.path.isdir(path_target) and len(glob.glob(os.path.join(path_target,'*.jpg')))>=num: continue
        if not os.path.isdir(path_target):
           os.mkdir(path_target)
        else:
            fns = glob.glob(path_target+'/*.jpg')
            if len(fns)>=num: continue
        shutil.copyfile(path_source, path_target+'.jpg')
        i_source = cv2.imread(path_source)
        i_source = cv2.cvtColor(i_source, cv2.COLOR_BGR2RGB)
        for c in range(num):
           i_target = image_preprocess(i_source, jitter=True, black_border=False, replace_green=DIR_BG, noise=True)
        #    i_target = clahe(i_target)
           i_target = (i_target*255).astype("uint8")
           i_target = cv2.cvtColor(i_target, cv2.COLOR_RGB2BGR)
           cv2.imwrite(os.path.join(path_target, str(c)+'.jpg'), i_target)
        # i_target = clahe(i_source)
        # i_target = (i_target*255).astype("uint8")
        # i_target = cv2.cvtColor(i_target, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(root_target, str(i)+'.jpg'), i_target)
        
        # cv2.imwrite(os.path.join(path_target, str(c)+'.jpg'), i_target)
        # print("Done: #", i)
    print("Voila.")

def clahe_batch():
    for i in tqdm(range(0, 756541)):
        path_source = os.path.join(root_source, str(i)+'.jpg')
        path_target = os.path.join(root_target, str(i)+'.jpg')
        if not os.path.isfile(path_source): continue
        i_source = cv2.imread(path_source)
        i_source = cv2.cvtColor(i_source, cv2.COLOR_BGR2RGB)
        i_target = clahe(i_source)
        i_target = (i_target*255).astype("uint8")
        i_target = cv2.cvtColor(i_target, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path_target, i_target)
    print("Voila.")
    
if __name__=="__main__":
    args = parser.parse_args()

    root_source = '/data/xgaze/448/eye_L_aug'
    root_target = '/data/xgaze/448/eye_L_aug_0415'
    DIR_BG = '/root/3d_to_2d/data/bg_imgs'
    DIR_ROT31 = '/root/3d_to_2d/data/rot31'

    manta_style(num=50, start=args.start, end=args.end)
    # clahe_batch()
#generate_bg() 