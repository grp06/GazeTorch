import os
import numpy as np
import random
from skimage.morphology import square, diamond, disk, octagon, star
from skimage.io import imsave
from scipy import ndimage

SHAPE_FUNCS = [square, diamond, disk, octagon, star]
def shape(shape_id, size):
    t = random.randrange(0, 5) if shape_id is None else shape_id
    if t==0:
        x = size
    elif t==4:
        x = size//3
    elif t==3:
        x = size//2
        return octagon(x, (x+1)//2)
    else:
        x = size//2
    return SHAPE_FUNCS[t](x)

def generate(target_dir, size):
    global counter
    counter = 0

    def save(image):
        global counter
        fn = os.path.join(target_dir, "{}.png".format(counter))
        imsave(fn, image * 255)
        counter += 1

    for shape_id in range(5):
        S = shape(shape_id, size)

        if shape_id == 1:
            continue

        if shape_id == 2:
            save(S)
            continue

        for pose_id in range(9):
            angle = pose_id * 10
            SR = ndimage.rotate(S, angle)
            save(SR)

if __name__=='__main__':
    generate("patterns250", 250)