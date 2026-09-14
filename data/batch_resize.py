'''
Example usage:
python batch_resize.py -i /data/xgaze/448/face_224 -o /data/xgaze/448/face_112 --suffix .jpg -s 112
'''
import os
import cv2
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="dir_in", type=str)
parser.add_argument("-o", dest="dir_out", type=str)
parser.add_argument("-s", "--size", type=int, default=112)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=-1)
parser.add_argument("--suffix", dest="suffix", type=str, default='.jpg')

def batch(dir_in, dir_out, suffix, size, start=0, end=-1):
    target_size = (size, size)
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)
    filelist = sorted(os.listdir(dir_in))
    if end==-1:
        end = len(filelist)
    for i in tqdm(range(start, end)):
        fn = filelist[i]
        if not fn.endswith(suffix): continue
        path_in = os.path.join(dir_in, fn)
        path_out = os.path.join(dir_out, fn)
        bgr = cv2.imread(path_in)
        out = cv2.resize(bgr, target_size)
        cv2.imwrite(path_out, out)
    print("Voila.")

if __name__=='__main__':
    args = parser.parse_args()
    batch(args.dir_in, args.dir_out, args.suffix, args.size, args.start, args.end)