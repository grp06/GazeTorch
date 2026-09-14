import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm

def main():
    root = '/data/MPIIGaze/MPIIFaceGaze_normalized'
    target_root = '/data/mpiiface/'
    target_dir = '224png'
    
    gt = {}
    gaze = np.zeros((0,2))
    headpose = np.zeros((0,2))
    landmarks = np.zeros((0,12))
    subject_ids = []
    counter = 0
    for sid, fn in enumerate(sorted(os.listdir(root))):
        if not fn.endswith('.mat'): continue
        if len(fn)!=7: continue
        print('#', counter, fn)
        mat = h5py.File(os.path.join(root, fn))
    
        imgs = mat['Data']['data']
        labels = mat['Data']['label']
        n = len(imgs)
        for i in tqdm(range(n)):
            img = imgs[i, ...].transpose(1, 2, 0)
            gaze = np.append(gaze, labels[i, :2])
            headpose = np.append(headpose, labels[i, 2:4])
            landmarks = np.append(landmarks, labels[i, 4:])
            img = cv2.resize(img, (224, 224))
            cv2.imwrite(os.path.join(target_root, target_dir, str(counter)+'.png'), img)
            subject_ids.append(sid)
            counter += 1

    gt['gaze'] = gaze.reshape(counter, -1)
    gt['headpose'] = headpose.reshape(counter, -1)
    gt['landmarks'] = landmarks.reshape(counter, -1)
    gt['subject_cid'] = subject_ids
    np.save(os.path.join(target_root, 'gt.npy'), gt)
    
    print('Voila. # of data =', counter)
    return counter
    
if __name__ == '__main__':
    main()