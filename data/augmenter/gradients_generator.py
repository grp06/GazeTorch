import numpy as np
import cv2
import os
from skimage.io import imsave

def generate_template_angle(angle, size=1000, black_ratio=0.25):
    TO_RADIAN = np.pi/180
    c = np.cos(angle * TO_RADIAN)
    s = np.sin(angle * TO_RADIAN)

    X,Y = np.meshgrid(range(size), range(size))
    V = X*c+Y*s
    cv2.normalize(V, V, 0, 1, cv2.NORM_MINMAX)
    V = np.clip(V, black_ratio, 1)
    cv2.normalize(V, V, 0, 1, cv2.NORM_MINMAX)

    return V

def generate_template_circle(target_path):
    canvas = np.zeros((512,512))
    c = (128, 256) # (y,x) coordinate of light center
    r = 256 # radius
    for y in range(512):
        for x in range(512):
            d = np.linalg.norm(np.array([y-c[0], x-c[1]]))
            canvas[y,x] = 1 - d / r
    canvas = np.clip(canvas, 0, 1)
    canvas *= 255
    cv2.imwrite(target_path, canvas)
    return canvas

def save(image, target_dir):
    global counter
    fn = os.path.join(target_dir, "{}.png".format(counter))
    imsave(fn, image * 255)
    counter += 1

def generate_template(target_dir):
    global counter
    counter = 0
    for angle in range(-180, 180, 10):
        SR = generate_template_angle(angle, 1000, 0.25) * 255
        # crop
        save(SR, target_dir)

if __name__=='__main__':
    # generate_template("gradients")
    generate_template_circle("overexposed/0.png")