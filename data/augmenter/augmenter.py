import os
import cv2
import random
import numpy as np
from scipy.stats import beta
from scipy import interpolate
from scipy.spatial.transform import Rotation

from data.augmenter.args import AugmenterArgsHandler
from data.augmenter.patcher import Patcher
from data.augmenter.transform2d import Transform2D
from data.augmenter.camera_noise import add_ycrcb_noise
from data.augmenter.bottom_mask import bottom_mask

import data.consts as consts

ROOT_BACKGROUND_IMAGES = "/data/mit_indoor/indoorCVPR_09/Images/"
ROOT_GLASSES_SAMPLES = '/data/Glasses_v5'

GLASSES_COLORS = ['black', 'blue', 'cyan', 'green', 'magenta', 'orange', 'purple', 'red', 'yellow']
GENERIC_FACE_MODEL_PATH = "tools/alex/face_generic.npy"


class AugmenterAssetBuffer:
    def __init__(self, root_materials="data/augmenter") -> None:
        self.root = root_materials
        self.root_gradients = os.path.join(root_materials, "gradients")
        self.root_patterns = os.path.join(root_materials, "patterns")

        self.patterns = self.load_materials("patterns")
        self.n_patterns = len(self.patterns)

        self.gradients = self.load_materials("gradients")
        self.n_gradients = len(self.gradients)
        self.GRADIENT_TEMPLATE_SIZE = 1000
        self.GRADIENT_BORDER_SIZE = 200

    def load_materials(self, var_name):
        var = []
        dn = getattr(self, "root_"+var_name)
        for fn in sorted(os.listdir(dn)):
            if not fn.endswith(".png"): continue
            path = os.path.join(dn, fn)
            bgr = cv2.imread(path)
            bgr = bgr.astype("float32") / 255
            var.append(bgr)
        print("[Augmenter]Loaded: ", var_name)
        return var
    
AUGMENTER_ASSET_BUFFER = AugmenterAssetBuffer()

class Augmenter:
    DEFAULT_ARGS_PATH_TRAIN = "config/augmenter/default_train.yaml"
    DEFAULT_ARGS_PATH_TEST = "config/augmenter/default_test.yaml"

    def __init__(self, root_materials="data/augmenter", root_imglib=ROOT_BACKGROUND_IMAGES, size=224,
        path_cfg=None, t2d_external=False):
        '''
        t2d_external: if True, skip generating random parameters for self.t2d per run.
            * only applies for "flip_mode=on" datasets, in flip-every-other-view contrastive scenario.
            * only works in face-as-input case.
        '''
        if root_materials != AUGMENTER_ASSET_BUFFER.root:
            self.asset_buffer = AugmenterAssetBuffer(root_materials)
        else:
            self.asset_buffer = AUGMENTER_ASSET_BUFFER

        self.root_gradients = os.path.join(root_materials, "gradients")
        self.root_patterns = os.path.join(root_materials, "patterns")

        # load from the asset buffer
        self.patterns = self.asset_buffer.patterns
        self.n_patterns = self.asset_buffer.n_patterns
        self.gradients = self.asset_buffer.gradients
        self.n_gradients = self.asset_buffer.n_gradients
        self.GRADIENT_TEMPLATE_SIZE = self.asset_buffer.GRADIENT_TEMPLATE_SIZE
        self.GRADIENT_BORDER_SIZE = self.asset_buffer.GRADIENT_BORDER_SIZE

        if root_imglib is not None and os.path.isdir(root_imglib):
            self.patcher = Patcher(root_imglib, size)
        else:
            self.patcher = None

        self.path_cfg = path_cfg
        self.arg_handler = AugmenterArgsHandler(path_cfg)
        self.args = self.arg_handler.args
        self.t2d = Transform2D(
            scale_min=self.args.jitter.scale_min,
            scale_max=self.args.jitter.scale_max,
            trans_max=self.args.jitter.trans_max,
            rotate_max=self.args.jitter.rotate_max
        )
        self.t2d_external = t2d_external

        # New Glasses Augmentation
        if self.args.glasses.switch is True:
            self.glasses_mask_dict, self.glasses_color_dict, self.glasses_lmk_dict = self.load_glasses()

    def load_glasses(self):
        '''
        [Joe Lee]
        Returns: 
              [Dict Name] : [key] | [value]
            - glasses_mask_dict: (pitch_bin, yaw_bin) | (list of glasses(1-10) mask corresponding to pitch/yaw bin)
            - glasses_color_dict: (glasses_mask_name) | (list of glasses masks with 7 colors)
            - glasses_lmk_dict: (subject_id) | (4,2) np.array storing lmks for glasses alignment
        '''
        glasses_mask_dict = {}
        glasses_color_dict = {}
        glasses_lmk_dict = {}

        glasses_mask_dir = 'glasses_masks'
        glasses_colors_dir = 'glasses_colors'

        glasses_mask_paths = os.listdir(f'{ROOT_GLASSES_SAMPLES}/{glasses_mask_dir}/')
        for mask_path in glasses_mask_paths:
            name_no_ext = mask_path.split('.png')[0] # -> e.g) 2210_p2_y4_g3_red
            parts = name_no_ext.split('_') # -> [2210, p2, y4, g3, red]
            glasses_mask_key = (int(parts[1][1]), int(parts[2][1])) # -> (2,4)

            mask = cv2.imread(os.path.join(ROOT_GLASSES_SAMPLES, glasses_mask_dir, mask_path))
            glasses_mask_dict.setdefault(glasses_mask_key, []).append((mask_path, mask))
        
        glasses_color_paths = os.listdir(f'{ROOT_GLASSES_SAMPLES}/{glasses_colors_dir}/')
        for glasses_color_path in glasses_color_paths:
            name_no_ext = glasses_color_path.split('.png')[0] # -> e.g) 2210_p2_y4_g3_red
            parts = name_no_ext.split('_') # -> [2210, p2, y4, g3, red]
            glasses_color_key = "_".join(parts[:-1]) # -> key: 2210_p2_y4_g3
            
            glasses_color = cv2.imread(os.path.join(ROOT_GLASSES_SAMPLES, glasses_colors_dir, glasses_color_path))
            glasses_color_dict.setdefault(glasses_color_key, []).append(glasses_color)
        
        glasses_lmk_paths = os.listdir(f'{ROOT_GLASSES_SAMPLES}/glasses_lmks/')
        for glasses_lmk_path in glasses_lmk_paths:
            name_no_ext = glasses_lmk_path.split('.npy')[0]
            subject_id = name_no_ext.split('_')[-1]
            lmk = np.load(os.path.join(ROOT_GLASSES_SAMPLES, 'glasses_lmks', glasses_lmk_path))
            glasses_lmk_dict.setdefault(subject_id, []).append(lmk)

        if glasses_mask_dict and glasses_color_dict and glasses_lmk_dict:
            print('[Augmenter]Loaded:  Glasses_v5')

        return glasses_mask_dict, glasses_color_dict, glasses_lmk_dict

    # def augment(self, input_single_rgb, jitter=False, color_jitter_gamma=False, color_jitter_spline=False, noise=False, normalize=True, mean=None, std=None,
    #             pattern=False, add_mask=False, rgb_mask=None, gradient=False, desat=False, blur=False, global_noise=False, demo=False,
    #             img_dict=None, replace_green=False, rgb_green=None, oneside_noise=False,
    #             glasses_landmarks=None, **kwargs):
    def augment(self, input_single_rgb, img_dict=None, 
                rgb_mask=None, rgb_green=None, demo=False,
                flip=False,
                subject_landmarks=None, subject_headpose=None,
                **kwargs):
        '''
        Aug 1 '22 update: img_dict={names: arrays}, rgb_dict={names: arrays} to toggle batch-augment mode.
            All images are augmented together to be spatially consistent.
            When they are not None, `input_single_rgb` should be `None` to avoid ambiguity.

        Oct 9 '22 update: I rely on a Patcher instance to replace green background & simulate glasses reflection.
        If using Patcher to simulate glasses, pass a standard 34x2 landmark matrix to `glasses_landmarks`, and pattern should be FALSE.

        Jan 18 '23 update: kwargs can only contain BOOL type arguments, where I will override the "switch" parameters accordingly.
            * see data.augmenter.AugmenterArgsHandler class for details

        Feb 8 '23 update: flip images here, instead of Dataset._load_rgb()

        Nov '25 update by Joe: glasses features

        Dec 24 '25 update: also returns which aug feature is switched on, under imgdict['switches']

        TODO: oneside_noise is applied only when "eye_L" and "eye_R" are both present in img_dict.
        '''

        if len(kwargs)>0:
            args = self.arg_handler.update_by_kwargs(True, **kwargs)
        else:
            args = self.arg_handler.args
        
        single_rgb = img_dict is None
        if single_rgb:
            img_dict = {"face": input_single_rgb}
        img_dict_keys = list(img_dict.keys())

        has_rgb = False
        for k in img_dict_keys:
            if k in consts.RGB_TYPES:
                has_rgb = True
                break

        do_noise = False
        switch_code = 0 # stores which feature is switched on (applicable to "mask" and "glasses")

        # [Joe Lee] - 2025/11/04 - Fixes to remove background first, then add glass augmentation afterwards
        if has_rgb and'face' in img_dict_keys and args.replace_green.switch and rgb_green is not None:
                rgb = img_dict['face']
                green_mask = rgb_green.astype(np.float32)/255
                img_dict['face'] = self.patcher.replace_green_background(rgb, green_mask)

        # [Joe Lee] - 2025/11/21 - New glasses augmentation with our custom frames 
        pitch_lst = [-30, -15, 0, 15, 30]
        yaw_lst_full = [-27, -18, -9, 0, 9, 18, 27]
        yaw_lst_masks = [0, 9, 18, 27]
        flip_glasses = False

        if all([has_rgb, 'face' in img_dict_keys, subject_landmarks is not None]):
            if (len(subject_landmarks) == 4 and random.random() < args.glasses.p):
                from data.augmenter.glasses_augmenter import add_glassesv2, get_glasses_pitch_yaw
                subject_img = img_dict['face'] 
                pitch_val, yaw_val = get_glasses_pitch_yaw(pitch_lst, yaw_lst_full, subject_headpose)

                # If yaw is negative, first change to positive to get mask/colors then flip 
                if yaw_val < 0: 
                    flip_glasses = True
                    yaw_val = yaw_val * -1
                
                pitch_bin, yaw_bin = pitch_lst.index(pitch_val), yaw_lst_masks.index(yaw_val)

                if pitch_bin in [0,1,2,3,4] and yaw_bin in [0,1,2,3]:
       
                    glasses_mask_fn, glasses_mask = random.choice(self.glasses_mask_dict[(pitch_bin, yaw_bin)])
                    
                    glasses_color_key = glasses_mask_fn.split("_mask")[0] # e.g) 2210_p2_y4_g3
                    glasses_lmk_key = glasses_mask_fn.split("_p")[0] # e.g) 2210

                    glasses_color = random.choice(self.glasses_color_dict[glasses_color_key])
                    glasses_lmk = self.glasses_lmk_dict[glasses_lmk_key]

                    glasses_scale = random.uniform(args.glasses.scale_min, args.glasses.scale_max)
                    frame_opacity = random.uniform(args.glasses.opacity_min, args.glasses.opacity_max)

                    if flip_glasses: # flip horizontally
                        glasses_mask = cv2.flip(glasses_mask, 1)
                        glasses_color = cv2.flip(glasses_color, 1)
                    
                    img_dict['face'], glasses_mask = add_glassesv2(subject_img, subject_landmarks, glasses_mask, glasses_color, glasses_lmk, glasses_scale, frame_opacity)

                    if args.glasses.reflection and random.random() < args.glasses.p_reflection:
                        glasses_opacity = [random.uniform(args.glasses.opacity_glass_left_min, args.glasses.opacity_glass_left_max), 
                                           random.uniform(args.glasses.opacity_glass_right_min, args.glasses.opacity_glass_right_max)]
                        img_dict['face'] = self.patcher.simulate_glasses_reflection(img_dict['face'], glasses_mask, glasses_opacity, args.glasses.select_frame)

                    switch_code += 2

        if has_rgb and args.color_jitter_spline.switch: # works for uint8 images
            for c in range(3):
                f = self.build_spline(3, args.color_jitter_spline.a)
                k = args.color_jitter_spline.k
                if k!=1:
                    k2 = k*k
                    e = k2 ** random.random() / k # [1/2, 2]
                    f = np.round(np.power(f,e) * 255).astype(int)
                else:
                    f = np.round(f * 255).astype(int)

                # apply the same transform to all RGB images
                for key, rgb in img_dict.items():
                    if key not in consts.RGB_TYPES: continue
                    # Also need uint8 images for jitter
                    rgb = rgb.astype(np.uint8)
                    rgb[..., c] = f[rgb[..., c]]
                    img_dict[key] = rgb

                del f

        for key, rgb in img_dict.items():
            if key=='facemap_v3': continue
            if rgb.dtype == 'uint8':
                rgb = rgb.astype("float32")
                M = np.amax(rgb)
                if M>1e-3:
                    rgb /= M
                img_dict[key] = rgb

        if has_rgb:
            # // Commented out as glasses_amin replaced glasses_landmarks
            # if 'face' in img_dict_keys and args.glasses_landmarks.switch and glasses_landmarks is not None and \
            #     random.random() < args.glasses_landmarks.p:
            #     rgb = img_dict['face']
            #     img_dict['face'] = self.patcher.simulate_glasses(rgb, glasses_landmarks)
            if 'face' in img_dict_keys and args.add_mask.switch and rgb_mask is not None and \
                random.random() < args.add_mask.p:
                opacity = self.random(args.add_mask.opacity_min, args.add_mask.opacity_max)
                color = np.array((random.random(), random.random(), random.random()))
                M = rgb_mask.astype(np.float32) / 255
                rgb = img_dict['face']
                rgb = rgb * (-M + 1) + rgb * M * (1-opacity) + M * opacity * color
                img_dict['face'] = rgb
                switch_code += 1

            if 'face' in img_dict_keys and args.bottom_crop.switch and \
                random.random() < args.bottom_crop.p:
                # opacity = self.random(args.bottom_crop.opacity_min, args.bottom_crop.opacity_max)
                color = np.array((0, 0, 0)) # color black
                
                rgb = img_dict['face']

                M = bottom_mask(rgb).astype(np.float32) / 255
                rgb = rgb * (-M + 1)
                img_dict['face'] = rgb

            if args.color_jitter_gamma.switch:
                # RGB & Brightness jitter (no saturation jitter yet)
                gamma_i = self.random(args.color_jitter_gamma.gamma_L_min, args.color_jitter_gamma.gamma_L_max)
                gamma_rgb = [self.random(args.color_jitter_gamma.gamma_C_min, args.color_jitter_gamma.gamma_C_max)
                    for _ in range(3)]
                
                for key, rgb in img_dict.items():
                    if key not in consts.RGB_TYPES: continue
                    expo = np.zeros_like(rgb)
                    for c in range(3):
                        expo[:, :, c] = gamma_rgb[c] + gamma_i
                    rgb = np.power(rgb, expo)
                    img_dict[key] = rgb

            if args.desat.switch and random.random() < args.desat.p: 
                k = random.random()
                for key, rgb in img_dict.items():
                    if key not in consts.RGB_TYPES: continue
                    # rgb = rgb.astype("float32")
                    # gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    # gray = np.repeat(gray[..., None], 3, 2)
                    gray = np.mean(rgb, axis=2, keepdims=True)

                    rgb = rgb * k + (1-k) * gray
                    img_dict[key] = rgb

            if args.noise.switch and random.random() < args.noise.p: # TODO: noise separate for each RGB image
                do_noise = True
                for key, rgb in img_dict.items(): # noise map varies in each image
                    if key not in consts.RGB_TYPES: continue
                    noise_std = self.random(args.noise.std_min, args.noise.std_max)
                    noise_map = np.random.normal(0, noise_std, rgb.shape)
                    rgb += noise_map
                    rgb = np.clip(rgb, 0., 1.)
                    img_dict[key] = rgb

            if args.pattern.switch and hasattr(self, "patterns"):
                # put a random colored box above.
                for key, rgb in img_dict.items(): # noise map varies in each image
                    if key not in consts.RGB_TYPES: continue
                    h, w, _ = rgb.shape
                    opacity = self.random(args.pattern.opacity_min, args.pattern.opacity_max)
                    patch = self.patterns[random.randrange(0, self.n_patterns)]
                    hp, wp, _ = patch.shape
                    # resize
                    ratioX = self.random(args.pattern.scale_min, args.pattern.scale_max)
                    ratioY = self.random(args.pattern.scale_min, args.pattern.scale_max)
                    wpr = int(wp*ratioX); hpr = int(hp*ratioY)
                    patch = cv2.resize(patch, (wpr, hpr))

                    # set a color
                    color = (random.random(), random.random(), random.random())

                    # translation & embed into image
                    cX = random.randrange(0, w)
                    cY = random.randrange(0, h)
                    xmin_I = max(cX - wpr // 2, 0)
                    xmax_I = min(cX + wpr - wpr // 2, w)
                    ymin_I = max(cY - hpr // 2, 0)
                    ymax_I = min(cY + hpr - hpr // 2, h)
                    xmin_P = -min(cX - wpr//2, 0)
                    xmax_P = wpr - max(cX + wpr - wpr // 2 - w, 0)
                    ymin_P = -min(cY - hpr // 2, 0)
                    ymax_P = hpr - max(cY + hpr - hpr // 2 - h, 0)
                
                    target_area_image = rgb[ymin_I:ymax_I, xmin_I:xmax_I, :]
                    target_area_patch = patch[ymin_P:ymax_P, xmin_P:xmax_P, :]
                    target_area = target_area_image * (- target_area_patch + 1) + \
                        target_area_image * target_area_patch * (1-opacity) + \
                        target_area_patch * opacity * color
                    rgb[ymin_I:ymax_I, xmin_I:xmax_I, :] = target_area
                    img_dict[key] = rgb

            if args.gradient.switch and hasattr(self, "gradients") and \
                random.random() < args.gradient.p:
                for key, rgb in img_dict.items():
                    if key not in consts.RGB_TYPES: continue
                    h, w, _ = rgb.shape
                    opacity = self.random(args.gradient.opacity_min, args.gradient.opacity_max)
                    color = np.array((random.random(), random.random(), random.random())) * args.gradient.color_value_max
                    xmin = random.randrange(0,self.GRADIENT_BORDER_SIZE)
                    xmax = random.randrange(self.GRADIENT_TEMPLATE_SIZE-self.GRADIENT_BORDER_SIZE, self.GRADIENT_TEMPLATE_SIZE)
                    ymin = random.randrange(0, self.GRADIENT_BORDER_SIZE)
                    ymax = random.randrange(self.GRADIENT_TEMPLATE_SIZE - self.GRADIENT_BORDER_SIZE,
                                            self.GRADIENT_TEMPLATE_SIZE)
                    gradient_id = random.randrange(0, self.n_gradients)
                    G = self.gradients[gradient_id][ymin:ymax, xmin:xmax, ...]
                    G = cv2.resize(G, (w,h), interpolation=cv2.INTER_NEAREST)
                      
                    rgb = rgb * (-G + 1) + rgb * G * (1-opacity) + G * opacity * color
                    img_dict[key] = rgb

            if args.blur.switch and random.random() < args.blur.p:
                for key, rgb in img_dict.items():
                    if key not in consts.RGB_TYPES: continue
                    sigma = self.random(args.blur.std_min, args.blur.std_max)
                    ksize = args.blur.kernel_size
                    rgb = cv2.GaussianBlur(rgb, (ksize,ksize), sigma)
                    img_dict[key] = rgb

        for key, img in img_dict.items():
            if key not in consts.RGB_TYPES: continue
            img_dict[key] = np.ascontiguousarray(img_dict[key])

        if args.jitter.switch:
            for type_set in (consts.FACE_TYPES, consts.EYE_L_TYPES, consts.EYE_R_TYPES):
                if not self.t2d_external and type_set==consts.FACE_TYPES: # for flip_every_other_view case, the flipped view reuses the T2D params
                    self.t2d.generate_params()
                for key, img in img_dict.items():
                    if not key in type_set: continue
                    img_dict[key] = self.t2d(img, self.t2d.buffer_params, is_rgb=key in consts.RGB_TYPES)

        if has_rgb and not do_noise:
            for key, img in img_dict.items():
                if key not in consts.RGB_TYPES: continue  # TODO: later, add salt&pepper noise to binary maps?
                if args.global_noise.switch and random.random() < args.global_noise.p:
                    global_noise_std = self.random(args.global_noise.std_min, args.global_noise.std_max)
                    noise_map = np.random.normal(0, global_noise_std, img.shape)
                    img += noise_map
                    img = np.clip(img, 0., 1.)
                    img_dict[key] = img

        if flip:
            for key, img in img_dict.items():
                if key not in consts.IMG_TYPES: continue
                img = img[:,::-1,:]
                if key.startswith('facemap'):
                    img = img[..., [2,3,0,1]] # switch L and R
                img = img.copy()
                img_dict[key] = img

        if has_rgb and 'face' in img_dict_keys and args.camera_noise.switch and random.random() < args.camera_noise.p:
            rgb = img_dict['face']
            chroma_cr_cb_bias = (args.camera_noise.chroma_cr_bias, args.camera_noise.chroma_cb_bias)
            rgb = add_ycrcb_noise(rgb, chroma_strength=args.camera_noise.chroma_strength, chroma_blotch_size=args.camera_noise.chroma_blotch_size, chroma_cr_cb_bias=chroma_cr_cb_bias,
                                  luma_sigma=args.camera_noise.luma_sigma)
            img_dict['face'] = rgb

        if demo:
            import matplotlib.pyplot as plt
            rgb = img_dict[img_dict_keys[0]]
            plt.imshow(rgb)
            plt.show()

        if args.normalize.switch:
            for key, rgb in img_dict.items():
                if key in consts.RGB_TYPES:
                    mean = args.normalize.mean
                    std = args.normalize.std
                    h, w, _ = rgb.shape
                    if mean is None:
                        mean_rgb = np.tile(np.mean(rgb, axis=(0, 1)), (h, w, 1))
                    else:
                        mean_rgb = mean
                    if std is None:
                        std_v = np.std(rgb, axis=(0, 1))
                        for i in range(3): 
                            if std_v[i]==0:
                                std_v[i]=1
                        std_rgb = np.tile(std_v, (h, w, 1))
                    else:
                        std_rgb = std
                    rgb -= mean_rgb 
                    rgb /= std_rgb
                    img_dict[key] = rgb

        if single_rgb:
            rgb = img_dict[img_dict_keys[0]]
            return rgb
        else:
            img_dict['aug'] = switch_code
            return img_dict

    def face_apply_mask(self, rgb, rgb_mask):
        '''
        A compact augmentation - just adding a mask. For SSL mode.
        rgb MUST be a face image.
        '''
        rgb = rgb.astype("float32")
        rgb /= np.amax(rgb)
        if random.random()<self.args.add_mask.p:
            opacity = self.random(self.args.add_mask.opacity_min, self.args.add_mask.opacity_max)
            color = np.array((random.random(), random.random(), random.random()))
            M = rgb_mask.astype("float32") / 255
            rgb = rgb * (-M + 1) + rgb * M * (1-opacity) + M * opacity * color
        return rgb

    @staticmethod
    def build_spline(n_nodes=3, k=0.5): # k in [0,1]
        x = np.linspace(0, 1, n_nodes+2)
        r = beta.rvs(1.1, 5, size=n_nodes+1)
        t = random.random() * k + 1 - k
        r = r*t + 1 - t
        cr = np.cumsum(r)
        cr /= cr[-1]
        y = np.zeros_like(x)
        y[1:] = cr

        x0 = np.linspace(0, 1, 256)
        f =  interpolate.PchipInterpolator(x, y)
        y0 = f(x0)
        return y0

    @staticmethod
    def random(v_min, v_max):
        return random.random() * (v_max-v_min) + v_min

def initialize_augmenter(split='test', augmenter_dir=consts.AUG_DIR_MATERIALS, augmenter_cfg_path=None, augmenter_asset_buffer=None) -> Augmenter:
    '''
    interface for dataset classes.
    split: 'train' or else
    augmenter_dir: dir storing material images.
    augmenter_cfg_path: path to the augmentation YAML config file.
    augmenter_options: a dict of customized arguments to be passed to the Augmenter during __getitem__().
        * Only bool type args are supported (xxx = True/False).
        * If specified, override the corresponding "switches" in augmentation YAML config
        * Add "augmenter_options" node in dataset_train/val.arg sections in Config files
    '''
    if augmenter_cfg_path is None:
        if split=='train':
            cfgpath = Augmenter.DEFAULT_ARGS_PATH_TRAIN
        else:
            cfgpath = Augmenter.DEFAULT_ARGS_PATH_TEST
    else:
        cfgpath = augmenter_cfg_path
    print("[Initialize_Augmenter]Loaded from", cfgpath)
    return Augmenter(augmenter_dir, path_cfg=cfgpath)

def process_getitem_values(dict_values:dict, list_dataset_all:list, switch_code:int):
    # order: mask, glasses
    if 'aug_mask' in list_dataset_all:
        dict_values['aug_mask'] = switch_code % 2
    switch_code //= 2
    if 'aug_glasses' in list_dataset_all:
        dict_values['aug_glasses'] = switch_code % 2
    return dict_values

if __name__=="__main__":
    rgb0 = cv2.imread("/data/xgaze/448/face_224/2.jpg")[...,::-1]
    rgb_mask = cv2.imread("/data/xgaze/448/facemask_224/2.png")
    rgb_green = cv2.imread("/data/xgaze/448/green/2.jpg")[...,::-1]

    a = np.load('/data/xgaze/448/lmk_224.npy', allow_pickle=True)
    glasses_landmarks = a[2]
    A = Augmenter("data/augmenter", path_cfg='/store/joelee217/GazeTorch/config/augmenter/temp_default_train_playground.yaml')
    import time
    ti = time.time()
    for i in range(100):
        rgb = rgb0.copy()
        # o = A.augment(rgb, rgb_mask=rgb_mask, color_jitter_spline=True, desat=False, noise=True, pattern=True, gradient=True,
        o = A.augment(rgb, rgb_mask=rgb_mask, rgb_green=rgb_green, normalize=False,
                      glasses_landmarks=glasses_landmarks, cam_idx=2) * 255
        cv2.imwrite('temp/%d.png' % i, o[:, :, ::-1])
    to = time.time()
    print("Time=", to-ti)
    pass
