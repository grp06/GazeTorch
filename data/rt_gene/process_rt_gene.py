import os
import re
import cv2
import sys
import random
import numpy as np
import utils as dpc
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.figure(figsize=(30, 30))

RT_GENE_DIR = "/data/rt_gene"


info_dict = {
    'fn': [],
    'gaze': [],
    'headpose': [],
    'subject_cid': [],
    'split': [],
    'fold_id': [],
    'glasses': []
}


if __name__ == "__main__":
    n_saved_data = 0
    n_saved_sample = 0
    im_root = os.path.join(RT_GENE_DIR, 'data')
    # Acquire the list of files
    file_names = os.listdir(im_root)
    file_names = [name for name in file_names if "_glasses" in name]
    file_names = list(filter(lambda x: '.tar' not in x, file_names))
    total = len(file_names)
    
    # Process each person's data
    for person_path in tqdm(file_names):
        label_path = os.path.join(im_root, person_path, "label_combined.txt")
        imgs_face = os.listdir(os.path.join(im_root, person_path, "original", "face"))
        imgs_face.sort(key = lambda x:int(x.split("_")[1]))

        pimgs_face = os.listdir(os.path.join(im_root, person_path, "inpainted", "face"))
        pimgs_face.sort(key = lambda x:int(x.split("_")[1]))
        with open(label_path) as infile:
            label_info = infile.readlines()
        
        plt.figure(figsize=(20, 20))
        for num, label in enumerate(label_info):
            label = re.split('\[| |,|\]', label.strip())
            for path in ['original', 'inpainted']:
                name_face = os.path.join(f"{person_path}/{path}/face/{imgs_face[num]}")
                person_name = person_path[0:4]
                head = dpc.GazeTo3d(list(map(eval, [label[3], label[5]])))
                gaze = dpc.GazeTo3d(list(map(eval, [label[9], label[11]])))
                gaze2d = dpc.GazeTo2d(gaze).astype('float32')
                gaze2d = gaze2d[::-1]
                head2d = dpc.GazeTo2d(head).astype('float32')
                head2d = head2d[::-1]

                face = cv2.imread(os.path.join(im_root, name_face))
                fn = str(n_saved_data) + str('_') + name_face.split('/')[-1].split('.')[0]

                if person_name in ["s001", "s002", "s008", "s010"]:
                    fold_id = 0
                    split = 'train'
                elif person_name in ["s003", "s004", "s007", "s009"]:
                    fold_id = 1
                    split = 'train'
                elif person_name in ["s005", "s006", "s011", "s012", "s013"]:
                    fold_id = 2
                    split = 'train'
                elif person_name in ["s000", "s014", "s015", "s016"]: # manually add s000 to validation fold, not based on http://phi-ai.buaa.edu.cn/Gazehub
                    fold_id = 3
                    split = 'val'
                else:
                    raise ValueError('not a valid person name')

                subject_cid = int(person_name.replace('s', ''))
                if 'inpainted' in path:
                    info_dict['glasses'].extend([0])
                else:
                    info_dict['glasses'].extend([1])
                info_dict['fn'].extend([fn])
                info_dict['gaze'].append(gaze2d+head2d)  # TODO: verify this
                info_dict['headpose'].append(head2d)
                info_dict['fold_id'].extend([fold_id])
                info_dict['subject_cid'].extend([subject_cid])
                info_dict['split'].extend([split])
                cv2.imwrite(os.path.join(RT_GENE_DIR, 'face_224', fn + '.png'), face)

                n_saved_data += 1
                if n_saved_sample < 100:
                    rand_num = random.randint(0, 100)
                    if rand_num <= 1:
                        n_saved_sample += 1
                        plt.subplot(10, 10, n_saved_sample)
                        draw_out = dpc.draw_gaze(face, head2d + gaze2d)
                        plt.imshow(draw_out[:, :, ::-1])
                        plt.title('[{}, {}]'.format(round((gaze2d[0]+head2d[0]) * 180/np.pi, 2), round((gaze2d[1]+head2d[1]) * 180/np.pi, 2)))
                        plt.axis('off')
                if n_saved_sample == 100:
                    n_saved_sample += 1
                    plt.savefig(os.path.join(RT_GENE_DIR, 'samples.png'))
                    print('\n samples saved ... \n')
                    

    info_dict['gaze'] = np.array(info_dict['gaze'])
    info_dict['headpose'] = np.array(info_dict['headpose'])
    print('num saved data: ', info_dict['gaze'].shape[0])
    np.save(os.path.join(RT_GENE_DIR, 'rt_gene'), info_dict)

        