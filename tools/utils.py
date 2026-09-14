from typing import Iterable
import cv2
import os
import math
import numpy as np
from skimage import io
import time


def args_to_dict(**kwargs):
    return kwargs

def shift_image_custom(img, shift_ratio_x, shift_ratio_y):
    """
    Shift image by specific ratios with white background.
    
    Args:
        img: OpenCV image (numpy array)
        shift_ratio_x: Shift as ratio of image width (positive = right, negative = left)
        shift_ratio_y: Shift as ratio of image height (positive = down, negative = up)
    
    Returns:
        Shifted image (numpy array)
    """
    height, width = img.shape[:2]
    
    # Convert ratios to pixels
    shift_x = int(width * shift_ratio_x)
    shift_y = int(height * shift_ratio_y)
    
    print(f"Shifting by: x={shift_x}px, y={shift_y}px")
    
    # Create white background
    shifted_img = np.ones_like(img) * 255
    
    # Calculate source and destination regions
    src_x_start = max(0, -shift_x)
    src_y_start = max(0, -shift_y)
    src_x_end = min(width, width - shift_x)
    src_y_end = min(height, height - shift_y)
    
    dst_x_start = max(0, shift_x)
    dst_y_start = max(0, shift_y)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    
    shifted_img[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        img[src_y_start:src_y_end, src_x_start:src_x_end]
    
    return shifted_img


def imageset_to_video(dir, path_out, fps=10, frame_size=(1920,1080)):
    writer = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, frame_size)

    for fn in sorted(os.listdir(dir)):
        if not fn.endswith('jpg'): continue

        img = io.imread(os.path.join(dir,fn))
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
        writer.write(img)

    writer.release()

def plot_points(image, points, print_id=True, path="temp.jpg", color=(255,255,255), radius=5):
    if image.dtype==float:
        image = (image*255).astype(np.uint8)
    points = points.astype(int)
    for i in range(len(points)):
        if print_id:
            cv2.putText(image, str(i), (points[i,0], points[i,1]), cv2.FONT_HERSHEY_PLAIN, 1, color)
        cv2.circle(image, (points[i,0], points[i,1]), radius, color, 1)
    if path is not None:
        io.imsave(path, image)
    return image

def detect_face_dlib(image):
    import dlib
    face_detector = dlib.get_frontal_face_detector()  # dlib takes BGR/grayscale input

    rects = face_detector(image, 1)

    if len(rects) == 0:
        return
    elif len(rects) > 1:
        # TODO: find the biggest head
        pass

    head_rect = rects[0]

    head_tl = head_rect.tl_corner()
    head_br = head_rect.br_corner()

    img_face = image[head_tl.y:head_br.y+1, head_tl.x:head_br.x+1, :]

    return head_rect, img_face

def plot3d(x,y,z,ax=None,c=None):
    import matplotlib.pyplot as plt
    # cmap = plt.cm.get_cmap(cmap)
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z,marker='.',c=c)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    return ax

def plot3d_arr(arr_nx3,ax=None,c=None):
    import matplotlib.pyplot as plt
    # cmap = plt.cm.get_cmap(cmap)
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(arr_nx3[...,0],arr_nx3[...,1],arr_nx3[...,2],marker='.',c=c)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    return ax

def pitchyaw_to_xyz(pitchyaw):
    pitch, yaw = pitchyaw
    return (math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(yaw) * math.cos(pitch))

def xyz_to_pitchyaw(xyz):
    if type(xyz)==np.ndarray:
        xyz /= np.linalg.norm(xyz)
    x,y,z = xyz
    return (math.asin(y), math.atan(x/z))

def draw_gaze(image_in, g3d, thickness=2, color=(0, 255, 255)):
    """Draw 3D gaze angle starting from image center."""
    pitchyaw = g3d if len(g3d)==2 else xyz_to_pitchyaw(g3d)
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    h, w, _ = image_out.shape
    length = w / 2.0
    pos = (int(w / 2.0), int(h / 2.0))

    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    # practical length of the vector = sqrt(1-z^2)
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out

def angle_cut(img_path, angle, shift, from_top=True):
    img = cv2.imread(img_path)
    slope = math.tan(math.radians(angle))
    h, w, d = img.shape
    y_0 = shift if from_top else h - shift
    for y in range(h):
        for x in range (w):
            if from_top and slope*x + y_0 > y:
                img[y, x] = [0, 0, 0]
            elif not from_top and slope*x + y_0 < y:
                img[y, x] = [0, 0, 0]
    cv2.imwrite("./out_cut/"+(img_path.split("/")[-1]), img)

def image_preprocess(rgb, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), jitter=False, plot=False, normalize=True,
                     black_border=False, replace_green=None, do_clahe=False, noise_std=None, transpose=False):
    rgb = rgb.astype("float32")
    rgb_max = np.amax(rgb)
    rgb /= rgb_max # then it's [0,1]

    def jitter_rgb(rgb):
        import random
        gamma_i = 1 + (random.random() - 0.5)  # [0.5-1.5]
        rgb = rgb ** gamma_i
        for c in range(3):
            gamma_rgb = 1 + (random.random() - 0.5) / 2  # [0.75-1.25]
            rgb[..., c] = rgb[..., c] ** gamma_rgb
        return rgb

    if replace_green and os.path.isdir(replace_green):
        import glob
        bg_imgs = glob.glob(os.path.join(replace_green, "*.*"))
        n_bg_imgs = len(bg_imgs)
        id_bg = np.random.randint(n_bg_imgs+1)
        if id_bg<n_bg_imgs:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            print(bg_imgs[id_bg])
            bg = cv2.imread(bg_imgs[id_bg])
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            bg = bg.astype(float)/255
            bg = jitter_rgb(bg)

            s_rgb = max(rgb.shape)
            h, w, _ = bg.shape
            x1 = np.random.randint(w-s_rgb)
            y1 = np.random.randint(h-s_rgb)
            s = np.random.randint(s_rgb, min(w-x1, h-y1))
            bg = bg[y1:y1+s, x1:x1+s, :]
            bg = cv2.resize(bg, (s_rgb, s_rgb))

            # scale = s_rgb/min(bg.shape[:2])
            # h, w, _ = bg.shape
            # bg = cv2.resize(bg, (int(w*scale), int(h*scale)))
            # offset = np.random.randint(max(bg.shape)-s_rgb+1)
            # if h>w:
            #     bg = bg[offset:offset+s_rgb, :, :]
            # else:
            #     bg = bg[:, offset:offset+s_rgb, :]

            l_green = np.array([70., 0.5, 0.])
            u_green = np.array([170., 1., 1.])
            mask = cv2.inRange(hsv, l_green, u_green)
            # mask = guidedFilter(rgb, mask, 5, 1e-6)

            mask = np.stack([mask,mask,mask], axis=2) / 255
            rgb = (-mask+1) * rgb + mask * bg
            # rgb = np.where(np.stack([mask,mask,mask], axis=2), bg, rgb)

    if jitter:
        import random
        gamma_i = 1 + (random.random() - 0.5) #[0.5-1.5]

        if plot:
            rgb0 = rgb.copy()
        rgb = rgb ** gamma_i
        for c in range(3):
            gamma_rgb = 1 + (random.random() - 0.5) / 2  # [0.75-1.25]
            rgb[..., c] = rgb[..., c] ** gamma_rgb

        if plot:
            import matplotlib.pyplot as plt
            plt.subplot(121)
            plt.imshow(rgb0)
            plt.subplot(122)
            plt.imshow(rgb)
            plt.show()

    if black_border: # and np.random.rand()>0.5:
        r_id = int(np.round(np.random.normal(15,7.5)))
        r_id = np.clip(r_id, 0, 30)
        shift = int(np.round(np.random.normal(0, 30)))

        from_top = np.random.rand()>0.5
        mask = io.imread("dataset/rot31/%02d.png" % r_id) / 255
        if from_top:
            shift+=224
        else:
            shift-=224
            mask = -mask + 1
        mask = mask[750-224+shift:750+224+shift, ...]
        rgb = rgb*mask

    if jitter:
        # random scaling & crop
        a = int(np.abs(np.round(np.random.normal(0, 20))))
        b = int(np.abs(np.round(np.random.normal(0, 20))))
        rgb = rgb[a:-b, a:-b, :]
        rgb = cv2.resize(rgb, (224,224))

    if do_clahe:
        rgb = clahe(rgb)

    if noise_std and noise_std>0:
        h,w,_ = rgb.shape
        noise_map = np.random.normal(0, noise_std, (h,w,3))
        rgb += noise_map
        rgb = np.clip(rgb, 0., 1.)

    if normalize:
        rgb -= mean
        rgb /= std

    if transpose: # HWC (TF, numpy) => CHW (PyTorch)
        rgb = rgb.transpose((2,0,1))

    return rgb

def rvec_to_euler(rvec): #return [p, y, r]
    rotation_mat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rotation_mat, np.array([[0.], [0], [0]])))
    _, _, _, _, _, _, pyr = cv2.decomposeProjectionMatrix(pose_mat)
    return pyr

def euler_to_rvec(pyr):
    pass

def pad_to_square(image):
    shape = np.r_[image.shape]
    pad_all = (shape.max() - shape[:2]).astype('uint32')
    pad_half = pad_all // 2
    img_pad = np.pad(
        image,
        ((pad_half[0], pad_all[0] - pad_half[0]), (pad_half[1], pad_all[1] - pad_half[1]), (0, 0)),
        mode='constant')
    return img_pad, pad_half

def rotate_points(pts, pitch, yaw=0, roll=0):
    # pts = Nx3
    from scipy.spatial.transform import Rotation as R
    rotation_matrix = R.from_euler('xyz', [pitch, yaw, roll], degrees=True).as_matrix()
    return np.dot(pts, rotation_matrix.T)

def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_xyz(a) if len(a) == 2 else a
    b = pitchyaw_to_xyz(b) if len(b) == 2 else b

    ab = np.sum(np.multiply(a, b))
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * 180.0 / np.pi

def remove_nans(L):
    for i in range(len(L)-1, -1, -1):
        if L[i]!=L[i]: del L[i]

def sigmoid(X):
   return 1/(1+np.exp(-X))

def clahe(rgb, output_format='uint8'):
    rgb = rgb.astype("float32")
    rgb_max = np.amax(rgb)
    rgb /= rgb_max # then it's [0,1]

    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    L = (hls[...,1]*255).astype('uint8')

    clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(4,4))
    L_eq = clahe.apply(L)

    # L_eq = cv2.equalizeHist(L)
    hls[...,1] = L_eq.astype(np.float32)/255.
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB) # 0~1
    if output_format=='uint8':
        rgb = (rgb * 255).astype('uint8')
    return rgb

def arg_loader(default_args:dict, **kwargs):
    args = default_args.copy()
    for key, value in kwargs.items():
        args[key] = value
    return args

def where(L:Iterable, obj):
    for i in range(len(L)):
        if L[i]==obj:
            return i
    return -1

def mix(normalized_rgb1, normalized_rgb2, alpha2=0.5):
    # normalized_rgb2 is a patch with partial black pixels
    # both images must have same size
    # both images must be float/double [0,1]
    return normalized_rgb1 * (-normalized_rgb2 + 1) + \
          (1 - alpha2) * normalized_rgb1 * normalized_rgb2 + \
          alpha2 * normalized_rgb2

def mix2(normalized_rgb1, normalized_rgb2, mask, alpha2=0.5):
    # normalized_rgb1, 2, mask are same size images
    # mask be binary 0/1
    # all images must be float/double [0,1]
    return (-mask+1) * normalized_rgb1 + \
        (1-alpha2) * mask * normalized_rgb1 + \
            alpha2 * mask * normalized_rgb2

def rotate_cv2(rgb, angle, interpolation=cv2.INTER_LINEAR):
    '''
    angle in degrees (counter-clockwise)
    interpolation: INTER_
        * NEAREST/LINEAR/CUBIC
    '''
    h,w = rgb.shape[:2]
    cX,cY = (w//2,h//2)
    R = cv2.getRotationMatrix2D((cX,cY), angle, 1.)

    cos_abs = abs(R[0,0])
    sin_abs = abs(R[0,1])

    out_w = int(h*sin_abs + w*cos_abs)
    out_h = int(h*cos_abs + w*sin_abs)

    R[0,2] += out_w/2 - w/2
    R[1,2] += out_h/2 - h/2

    rotated = cv2.warpAffine(rgb, R, (out_w,out_h), flags=interpolation)
    return rotated

def rotate_vector(v, angle):
    ang = angle * np.pi / 180
    c = np.cos(ang)
    s = np.sin(ang)
    R = np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])
    return np.dot(R, v)

def palette(n):
    '''
    generate N colors on HSV (S=V=100%, H uniformly distributed)
    '''
    c = np.random.rand(n, 3)
    # c[:, 0] *= 360 # H
    c[:, 0] = np.linspace(0, 360, n+1)[:n]
    c[:, 1] *= 0.6 # L
    c[:, 1] += 0.2
    c[:, 2] *= 0.6 # S
    c[:, 2] += 0.4
    
    c = c[None, ...].astype(np.float32)
    rgb = cv2.cvtColor(c, cv2.COLOR_HLS2RGB).squeeze()
    return rgb

def visualize_facemap(facemap, face=None, debug=False):
    '''
    by Mehdi. Visualize a 8(10)xHxW facemap given by a GazeTorch segmentation head.
    face: background face image, if applicable (should be same sized).
    debug: if True, print some simple statistics for debugging.
    '''
    import torch
    d, h, w = facemap.shape
    rgb_facemap = np.zeros((h, w, 3))

    eyelid = d > 8
    c_eye = 3 if eyelid else 2
    c_iris = 2

    # 8 channels, each part 2 channels
    c = 0
    for i in range(4):
        # for each 2 channel first apply softmax, then last channel (channel 1) is probability of 1
        rgb = np.zeros((h, w, 3))
        c_current = c_eye if i%2==0 else c_iris
        seg = facemap[c:c+c_current]
        c += c_current

        seg = torch.nn.functional.softmax(seg, dim=0).cpu().detach().numpy()
        if debug:
            print("[Visualizer]max of channel #1 (eye/iris):", np.amax(seg[1:,...]))
        if i % 2 == 0:
            rgb[:, :, 1] = seg[1, ...] # eye
            if eyelid:
                if debug:
                    print("[Visualizer]max of eyelid:", np.amax(seg[2:,...]))
                rgb[:, :, 2] = seg[2, ...] # eyelid
        else:
            rgb[:, :, 0] = seg[1, ...] # iris
        
        rgb_facemap += rgb

    if face is None:
        result = rgb_facemap
    else:
        if face.dtype=='uint8':
            face = face.astype(float)/255
        result = mix(face, rgb_facemap, 0.25)
    result = (result*255).astype('uint8')
    if debug:
        print("[Visualizer]max of result:", np.amax(result))
    return result

def linear_regression_2d(pred_2d, gt_2d, print_output=False):
    from scipy.stats import linregress
    x_reg = linregress(pred_2d[:,0], gt_2d[:,0])
    y_reg = linregress(pred_2d[:,1], gt_2d[:,1])
    usp_2d = (x_reg.slope, x_reg.intercept, y_reg.slope, y_reg.intercept)

    if print_output:
        print("[Calibration] kx =", usp_2d[0])
        print("[Calibration] bx =", usp_2d[1])
        print("[Calibration] ky =", usp_2d[2])
        print("[Calibration] by =", usp_2d[3])

    return usp_2d

# def linear_regression_1d(pred, gt, weight=None, print_output=False):
#     from sklearn.linear_model import LinearRegression as LR
#     lr = LR().fit(pred[:, None], gt, sample_weight=weight)
#     usp = (lr.coef_[0], lr.intercept_)

#     if print_output:
#         print("[Calibration] k =", usp[0])
#         print("[Calibration] b =", usp[1])

#     return usp

def linear_regression_1d(pred, gt, weight=None, print_output=False):
    '''
    pred: N x 1 
    gt: N x 1
    weight: N x 1
    '''
    # need column 1 for intercept for matrix mult -> X: N x 2
    X = np.column_stack([pred, np.ones_like(pred)])

    # multiply by weights
    if weight is not None:
        W = np.diag(weight)
        # Closed form for LR -> 2 x 1
        betas = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ gt
    else:
        betas = np.linalg.inv(X.T @ X) @ X.T @ gt

    usp = (betas[0], betas[1])

    if print_output:
        print("[Calibration] k =", usp[0])
        print("[Calibration] b =", usp[1])

    return usp
