'''
MPIris class
Official model description
Input(s)
* Image of proportionally cropped left eye with
eyebrow (or horizontally flipped right eye) with a
25% margin on each side and size 64x64
Output(s)
* Eye with eyebrow region surface represented
as 71 2D landmarks flattened into a 1D tensor: (x1,
y1), (x2, y2), ...; x- and y-coordinates follow the
image pixel coordinates.
---Zhenhao's Notes: Apply the model on left (LHS in image, not physical left eye) eye images + right eye flipped images!
* Iris surface represented as 5 2D landmarks (1 for
pupil center and 4 for iris contour) also
represented as flattened 1D tensor.

===========================

iris_cdf(): iris segmentation by Zhenhao
'''
import numpy as np
import cv2
import os

from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
from skimage.morphology import convex_hull_image

from tools.utils import pad_to_square

EYE_LD_IDS = (0,1,2,3,4,5,6,7,8,15,14,13,12,11,10,9)
class MPIris:
    def __init__(self, model_dir='cv/modules'):
        import tensorflow as tf
        model_path = os.path.join(model_dir, "iris_landmark.tflite")
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.in_idx = input_details[0]['index']
        self.out_idx_eye = output_details[0]['index']
        self.out_idx_iris = output_details[1]['index']

    def __call__(self, rgb, demo_path=None):
        '''
        :param rgb:  RGB Image (uint8/float)
        :param demo_path: save result at demo_path if not None.
        :return: a (2) tuple
            * eye_xy: 16x2 eye landmarks surrounding the eye (left corner - bottom line - right corner - top line)
            * iris_xy: 5x2 iris landmarks (centre, east, north, west, south)
        '''

        h, w, _ = rgb.shape
        max_dim = max(h,w)

        # pad image to square
        rgb = rgb.astype("float32")
        rgb = cv2.normalize(rgb, rgb, 0, 1, cv2.NORM_MINMAX)
        img_pad, pad_half = pad_to_square(rgb)
        img_pad = cv2.resize(img_pad, (48, 48))
        X = cv2.copyMakeBorder(img_pad, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=[0.5, 0.5, 0.5])

        # run tflite model
        self.interpreter.set_tensor(self.in_idx, X[None])
        self.interpreter.invoke()
        eye_xyz = self.interpreter.get_tensor(self.out_idx_eye).reshape((-1, 3))  # 71x3, but we use first 16 landmarks only
        iris_xyz = self.interpreter.get_tensor(self.out_idx_iris).reshape((-1, 3))  # 5x3

        # transform the coordinates back to original coordinate system
        scale = max_dim / 48
        trans = -pad_half[::-1].astype(int)
        eye_xy = ((eye_xyz[:,:2] - 8) * scale + trans).astype(int)
        iris_xy = ((iris_xyz[:,:2] - 8) * scale + trans).astype(int)

        if demo_path:
            img_out = cv2.cvtColor(np.uint8(rgb * 255.0), cv2.COLOR_RGB2BGR)

            # 5 iris landmark points, 4 + 1center
            for l in range(5):
                cv2.circle(img_out, (int(iris_xy[l, 0]), int(iris_xy[l, 1])), 2, (0, 0, 255), -1)
                cv2.putText(img_out, str(l), (int(iris_xy[l, 0]), int(iris_xy[l, 1])), cv2.FONT_HERSHEY_PLAIN, 1,
                            (255, 255, 0))

            for l in range(16):
                cv2.circle(img_out, (int(eye_xy[l, 0]), int(eye_xy[l, 1])), 2, (0, 255, 0), -1)
                cv2.putText(img_out, str(l), (int(eye_xy[l, 0]), int(eye_xy[l, 1])), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,255,255))

            # img_out = cv2.resize(img_out, (200, 200))  # resize to 200x200 to match original size 150x90
            cv2.imwrite(demo_path, img_out)

        return eye_xy[EYE_LD_IDS,:], iris_xy

'''
Pupil detection based on NUMPY and OPENCV.
'''

def diamond(radius, dtype=np.float32):
    L = np.arange(0, radius * 2 + 1)
    I, J = np.meshgrid(L, L)
    return np.array(np.abs(I - radius) + np.abs(J - radius) <= radius,
                    dtype=dtype)

def remove_black(rgb, opening_kernel_size=13):
    mask = (np.amax(rgb, axis=2)<5).astype('uint8') * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size)))
    rgb[mask>0] = 255
    return rgb

def locate_range(vector, threshold=0.7):
    vector = vector / np.sum(vector)
    threshold_cum = (1-threshold)/2
    x = 0; s = 0
    while s<threshold_cum:
        s+=vector[x]
        x+=1
    x1 = x
    x = len(vector)-1; s=0
    while s<threshold_cum:
        s+=vector[x]
        x-=1
    x2 = x
    return (x1+x2)/2

def iris_cdf(rgb, landmarks=None, lum_threshold=0.45, remove_reflection=True, figure=False,
                    range_threshold=0.7, debug=False, opening_kernel_size=13, gaussian_sigma=15):
    if debug:
        import time
        t_start = time.time()

    rgb = remove_black(rgb)
    Y = rgb[...,0] * 0.299 + rgb[...,1] * 0.587 + rgb[...,2] * 0.114
    Y = Y.astype(np.float32)
    h,w = Y.shape
    Y0 = Y.copy()
    if remove_reflection:
        gauss_kernel = cv2.GaussianBlur(diamond(3), (7,7), 2)
        Y = cv2.morphologyEx(Y, cv2.MORPH_OPEN, gauss_kernel)
    if landmarks is None:
        mask_poly_bin = mask_poly = None
        pupil_r = w // 12
        Y_ineye = Y
    else:
        xmin = np.min(landmarks[:,0])
        xmax = np.max(landmarks[:,0])
        pupil_r = max((xmax-xmin)//6,2)

        mask_poly = np.full((h, w), 0, np.uint8)
        cv2.fillPoly(mask_poly, [landmarks.astype(np.int32)], 255)
        mask_poly_bin = mask_poly.copy()

        # mask_poly = mask_poly.astype(np.float16)
        mask_poly = cv2.dilate(mask_poly, cv2.getStructuringElement(cv2.MORPH_RECT,(pupil_r//2, pupil_r//2)))
        mask_poly = cv2.GaussianBlur(mask_poly, (pupil_r*2-1, pupil_r*2-1), 15) 
        mask_poly = np.float32(mask_poly/255)

        Y += (1-mask_poly)*128
        Y = np.clip(Y, 0, 255)
        Y_ineye = np.ravel(Y[mask_poly_bin>0])

    # pupil_r = max(pupil_r0, pupil_r)
    y_t = np.quantile(Y_ineye, lum_threshold)
    # y_t = np.quantile(Y[Y<255], lum_threshold)
    if debug:
        print("Threshold_Y = {}".format(y_t))
    mask = np.uint8(Y<=y_t)

    mask0 = mask.copy()
    # mask_e = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (pupil_r//5, pupil_r//5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(opening_kernel_size,opening_kernel_size)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    # if np.sum(mask_e)>0: mask=mask_e
    if np.sum(mask)==0:
        y_t = np.quantile(Y_ineye, 0.1)
        mask = np.uint8(Y<=y_t)

    ret, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    mask = np.float32(labels == 1)
    pupil_r_odd = pupil_r // 2 * 2 + 1
    mask = cv2.GaussianBlur(mask, (pupil_r_odd, pupil_r_odd), gaussian_sigma)

    sum_x = np.sum(mask, axis=0)
    sum_y = np.sum(mask, axis=1)

    centroid_x = locate_range(sum_x, range_threshold)
    centroid_y = locate_range(sum_y, range_threshold)

    # binarize
    mask_poly = mask_poly_bin
    mask = (mask > 0.2).astype(np.float32)
    mask = convex_hull_image(mask).astype(np.float32)*255

    if debug:
        t_end=time.time()
        print("Elapsed time = {}".format(t_end-t_start))

    if figure:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('gray')
        plt.figure()
        plt.subplot(231)
        plt.imshow(Y0/255., cmap=cmap)
        plt.subplot(232)
        plt.imshow(Y/255., cmap=cmap)
        plt.subplot(233)
        plt.imshow(mask_poly_bin if mask_poly_bin is not None else np.zeros_like(mask), cmap=cmap)
        plt.subplot(234)
        plt.imshow(np.float32(mask0), cmap=cmap)
        plt.subplot(235)
        plt.imshow(np.float32(mask), cmap=cmap)
        plt.subplot(236)
        plt.imshow(Y/255., cmap=cmap)
        plt.scatter(centroid_x, centroid_y, c='r', marker="*")
        plt.show()

    return Y, mask, centroid_x, centroid_y, mask_poly


def get_best_params(rgb, landmarks=None, remove_reflection=True, range_threshold=0.7):
    # score = intersection + circularity 
    total_score = []
    lum_threshold_range = np.arange(0.05, 0.6, 0.1)
    gaussian_kernel_range = [15]
    opening_kernel_range = [13]
    params = []

    for lum_thresh in lum_threshold_range:
        gaussian_kernel = np.random.choice(gaussian_kernel_range)
        opening_kernel = np.random.choice(opening_kernel_range)
        Y, mask, _, _, mask_poly = iris_cdf(rgb, landmarks=landmarks, lum_threshold=lum_thresh, remove_reflection=remove_reflection, figure=False,
                    range_threshold=range_threshold, debug=False, opening_kernel_size=opening_kernel, gaussian_sigma=gaussian_kernel)

        if np.sum(mask) < 1:
            continue

        intersection_score = np.sum(cv2.bitwise_and(mask.astype('uint8'), mask_poly.astype('uint8'))) / np.sum(mask_poly)
        if intersection_score < 0.1:
            continue

        # get circularity of the mask
        cnts, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(cnts) == 0:
            continue
        cnt = sorted(cnts, key=cv2.contourArea)[0]
        area = cv2.contourArea(cnt)
        if area < 0.01:
            continue
        convexHull = cv2.convexHull(cnt)
        convexHull_perimeter = cv2.arcLength(convexHull, True)
        roundness = 4 * np.pi * area / (convexHull_perimeter ** 2)
        total_score.extend([intersection_score + roundness])
        params.extend([(lum_thresh, gaussian_kernel, opening_kernel)])
          
    if len(total_score) == 0:
        return (0.45, 15, 13)
    else:
        best_index = total_score.index(max(total_score))
        return params[best_index]

def iris_cdf_with_search(rgb, iris_detector:MPIris=None, landmarks=None):
    '''
    A wrapper function of Mehdi's parameter search
    '''
    if landmarks is None:
        landmarks, _ = iris_detector(rgb)
    best_params = get_best_params(rgb, landmarks)
    return iris_cdf(rgb, landmarks, figure=False, lum_threshold=best_params[0], gaussian_sigma=best_params[1], opening_kernel_size=best_params[2])


if __name__ == '__main__':
    from glob import glob
    id = MPIris()
    import time
    img_dirs = glob(".\\iris_challenge\\*.jpg")
    #img_dirs = [".\\iris_challenge\\00003_00095.jpg"]
    for img_dir in img_dirs: 
        rgb = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)
        rgb = cv2.flip(rgb, 1) # flip for right eyes
        reye_landmarks, _ = id(rgb)
        Y, mask, _, _, mask_poly = iris_cdf(rgb, reye_landmarks)
        s = time.time()
        best_params = get_best_params(rgb, reye_landmarks)
        print(time.time()-s)
        Y_changed, mask_changed, _, _, mask_poly_changed = iris_cdf(rgb, reye_landmarks, figure=False, lum_threshold=best_params[0], gaussian_sigma=best_params[1], opening_kernel_size=best_params[2])
        rgb_1 = np.copy(rgb)
        res = np.concatenate([cv2.cvtColor(rgb_1[:, :, ::-1],cv2.COLOR_BGR2GRAY), Y, mask, mask_poly], axis=1).astype('uint8')
        rgb_1[mask == 255] = 0
        res = np.concatenate([cv2.merge([res, res, res]), rgb_1[:, :, ::-1]], axis=1).astype('uint8')

        rgb_2 = np.copy(rgb)
        res2 = np.concatenate([cv2.cvtColor(rgb_2[:, :, ::-1],cv2.COLOR_BGR2GRAY), Y_changed, mask_changed, mask_poly_changed], axis=1).astype('uint8')
        rgb_2[mask_changed == 255] = 0
        res_changed = np.concatenate([cv2.merge([res2, res2, res2]), rgb_2[:, :, ::-1]], axis=1).astype('uint8')

        res_tot = np.concatenate([res, res_changed], axis=0)

        name_img = img_dir.split('\\')[-1].split('.')[0]

        cv2.imwrite('.\\res\\{}.png'.format(name_img), res_tot)
    
