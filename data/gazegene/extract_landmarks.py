'''
Author: Joe Lee 
Extract landmarks for images (image_224) using mediapipe.
* stores individual .npy dicts corresponding to each ___.png file for images
* Note that for the Dataset classes, individual .npy files are not used.
* Instead, a single lmk.npy file -> [key: image_id value: list of coords (.npy file)]
* is used to parse/process landmarks for augmenting glasses.
* Use lmk_dir_to_npy_conversion.py to process directory of .npy files to a single .npy file.
* Use parallel threadpool from multiprocessing library instead of splitting by index for convenience.
* ~ 4 hours to process 1M images with 72 workers.
'''
import os
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
import concurrent.futures
mp_face_mesh = mp.solutions.face_mesh
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class GLASSES:
    class RIGHT:
        test1 = [464, 443, 446, 450]
        test2 = [443, 450]
    class LEFT:
        test1 = [244, 223, 226, 230]
        test2 = [223, 230]
    ALL_1 = RIGHT.test1 + LEFT.test1
    ALL_2 = RIGHT.test2 + LEFT.test2

def get_landmark(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmark_dict = {}
    # Run face mesh on the image
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            print("No face landmarks detected.")
        else:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = image.shape
            landmark_indices = GLASSES.ALL_2 #OUTER.LEFT.CONTOUR + OUTER.RIGHT.CONTOUR #GLASSES.ALL_1 # [1, 468, 473]  # Nose tip and eyes (if refine_landmarks=True)

            for idx in landmark_indices:
                lm = face_landmarks.landmark[idx]
                landmark_dict[idx] = (lm.x, lm.y)
            
    return landmark_dict

def process_image(args):
    file_path, folder_path = args
    img_path = os.path.join(folder_path, file_path)
    
    try:
        # Extract ID from filename (assuming '123.png')
        file_id = int(os.path.splitext(file_path)[0])
        lmk = get_landmark(img_path)
        
        if lmk: # Handles both empty dict and None
            return (file_id, lmk)
        return (file_id, None)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return (None, None)

# Script used for ZeroGaze
def zerogaze_processing():
    ROOT = '/data/sfhq_fluxdev_gazezero'
    FACE_DIR = os.path.join(ROOT, 'face_224')
    
    gt_copy = np.load(os.path.join(ROOT, "gt_zerogaze.npy"), allow_pickle=True).item()
    id_list = gt_copy['id']
    id_len = len(id_list)

    id_to_idx = {id_val: i for i, id_val in enumerate(id_list)}
    save_npy = [None] * id_len

    files = [f for f in os.listdir(FACE_DIR) if f.endswith(".png")]
    task_args = [(f, FACE_DIR) for f in files]

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(process_image, task_args, chunksize=50), 
            total=len(task_args),
            desc="Detecting Landmarks"
        ))

    for file_id, landmarks in results:
        if file_id is not None and landmarks is not None:
            if file_id in id_to_idx:
                save_npy[id_to_idx[file_id]] = landmarks
    
    gt_copy['lmk'] = save_npy
    print("New gt_copy saved!")
    print(gt_copy.keys())

    save_dir = os.path.join(ROOT, 'gt_zerogaze_v2.npy')
    np.save(save_dir, gt_copy)

# Script used for GazeGene
def gazegene_processing():
    # replace paths as necessary
    src_path = '/store/GazeGene/face_224'
    save_path = '/home/joelee217/lmk_224_parallel'

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in os.listdir(src_path) if f.endswith(".png")]

    with concurrent.futures.ThreadPoolExecutor(max_workers=72) as executor:
        list(executor.map(lambda f: process_image(f, src_path, save_path), files), total=len(files))

    print(f"Data saved in directory '{save_dir}'.")

