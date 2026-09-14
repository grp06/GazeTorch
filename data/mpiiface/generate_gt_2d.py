import numpy as np
import os
from scipy.io import loadmat

ROOT_LABELS = '/data/MPIIGaze/260306_MPIIGaze/Label/'
ROOT_MATS = ''
N = 45000

# PATH_OUTPUT = '/data/MPIIGaze/gt_2d.npy'
PATH_OUTPUT = 'temp/mpii2d_251227/gt_2d_raw.npy'
result = np.zeros((N, 2))

label_lines = []
label_fns = sorted(os.listdir(ROOT_LABELS))
print(label_fns)
for label_fn in label_fns:
    fn = os.path.join(ROOT_LABELS, label_fn)
    with open(fn) as f:
        lines = f.readlines()

    label_lines += lines[1:] # 45K lines

screen_params = []
for i in range(15):
    mat_path = f"/data/MPIIGaze/MPIIFaceGaze_original/p{i:02d}/Calibration/screenSize.mat"
    mat = loadmat(mat_path)
    screen_params.append(mat)

    print("#", i)
    print("W:", mat['width_mm'][0][0])
    print("H:", mat['height_mm'][0][0])

invalid_ids = []
for i in range(N):
    pid = i // 3000

    label = label_lines[i]
    target_2d = label.split(' ')[-1]

    g2d_px = np.fromstring(target_2d, sep=',')
    screen_param = screen_params[pid]

    if g2d_px[0] > screen_param['width_pixel'][0][0] or g2d_px[1] > screen_param['height_pixel'][0][0]:
        print(f"Outlier! #{i}: {g2d_px}")
        invalid_ids.append(i)

    result[i, 0] = (g2d_px[0] / screen_param['width_pixel'][0][0] - 0) * screen_param['width_mm'][0][0]
    result[i, 1] = (g2d_px[1] / screen_param['height_pixel'][0][0] - 0) * screen_param['height_mm'][0][0]
    
# np.save(PATH_OUTPUT, result)
np.save('/data/mpiiface/invalid_ids.npy', np.array(invalid_ids))
# print("Voila.")