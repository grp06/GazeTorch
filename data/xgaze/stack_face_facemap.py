import numpy as np
import cv2

def contour(im, color=(255, 0, 0), thickness=2):
    ret,thresh = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    z = np.zeros((im.shape[0], im.shape[1], 3))
    cv2.drawContours(z,contours,-1, color, thickness)
    return z

def gray_to_rgb(G):
    return np.stack([G,G,G], axis=2)

def stack_face_facemap(face, facemap):
    # Given face image and corresponding facemap, encode them into a single RGB for visualization.
    # face: BGR image given by cv2.imread(face.jpg)
    # facemap: 4-channel facemap given by cv2.imread(facemap.png)
    
    mask_leye = facemap[:, :, 0] 
    mask_liris = facemap[:, :, 1] 
    mask_reye = facemap[:, :, 2] 
    mask_riris = facemap[:, :, 3]

    I = mask_liris + mask_riris
    E = mask_leye + mask_reye

    contour_I = contour(I, (0, 255, 0))
    contour_E = contour(E, (255, 0, 255))
    contour_I_bin = contour_I[...,1] / 255
    contour_I_bin = gray_to_rgb(contour_I_bin)
    C = contour_E * (-contour_I_bin+1) + contour_I
    C_bin = np.amax(C, axis=2) / 255
    C_bin = gray_to_rgb(C_bin)

    result_bgr = face * (-C_bin+1) + face * 0.5 * C_bin + C * 0.5 * C_bin
    return result_bgr

if __name__=='__main__':
    face = cv2.imread(r"D:\data\xgaze\220823\0.jpg")
    facemap = cv2.imread(r"D:\data\xgaze\220823\0m.png", cv2.IMREAD_UNCHANGED)
    # G = facemap[...,0] + facemap[..., 2]
    # contour(G)
    z=stack_face_facemap(face, facemap)
    cv2.imwrite(r"D:\data\xgaze\220823\0t.jpg", z)