# standard imports 
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys, os
import concurrent.futures

IMG_RAW_DIR = Path("/data/sfhq_fluxdev_gazezero_v2/results_sfhq_fluxdev__gazezero_noglasses/")
IMG_224_DIR = Path("/data/sfhq_fluxdev_gazezero_v2/gazezero_vanilla_224/")
IMG_NPY_DIR = Path('/data/sfhq_fluxdev_gazezero_v2/gazezero_vanilla_gt')


# imports for lmk generation
# import mediapipe as mp
# mp_face_mesh = mp.solutions.face_mesh

# class GLASSES:
#     class RIGHT:
#         test1 = [464, 443, 446, 450]
#         test2 = [443, 450]
#     class LEFT:
#         test1 = [244, 223, 226, 230]
#         test2 = [223, 230]
#     ALL_1 = RIGHT.test1 + LEFT.test1
#     ALL_2 = RIGHT.test2 + LEFT.test2

# landmark_indices = GLASSES.ALL_2 #OUTER.LEFT.CONTOUR + OUTER.RIGHT.CONTOUR #GLASSES.ALL_1 # [1, 468, 473]  # Nose tip and eyes (if refine_landmarks=True)

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh_session = mp_face_mesh.FaceMesh(
#     static_image_mode=True,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5
# )

# imports for headpose estimation
from cv.headpose import HeadposeDetector
from cv.face_detector import FaceDetector


# ---------------------------------------------------------------------
# 1. Extract landmarks 
def get_landmark(img):
    h, w, _ = img.shape
    landmark_dict = {}
    # results = face_mesh_session.process(img)

    # if results.multi_face_landmarks is None:
    #     return landmark_dict

    # face_landmarks = results.multi_face_landmarks[0]    

    # for idx in landmark_indices:
    #     lm = face_landmarks.landmark[idx]
    #     landmark_dict[idx] = (lm.x, lm.y)
            
    return landmark_dict

# 2. Extract Headposes
def get_headpose(img):
    try:
        headpose = poser(img)
        return np.array([headpose[0], headpose[1], 0.0])

    except Exception:
        return np.array([None, None, None])

def process_image(img_path, req_lmks=False):
    lmk, hp = None, None

    img = cv2.imread(str(img_path))
    if img is None:
        return None, lmk, hp
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_cropped, _ = face_cropper.run_and_crop(img_rgb)
    img_224 = cv2.resize(img_cropped, (224, 224))

    if req_lmks:
        lmk = get_landmark(img_224)
    
    hp = get_headpose(img_224)

    # e.g) /data/dir../10292.png -> stem: 10292
    img_id = str(img_path.stem).split("_")[0]
    save_path = IMG_224_DIR / f"{img_id}.png"
    cv2.imwrite(str(save_path), cv2.cvtColor(img_224, cv2.COLOR_RGB2BGR))

    return img_id, lmk, hp


def init_worker():
    global poser, face_cropper
    poser = HeadposeDetector(model_dir='/store/joelee217/GazeTorch/cv/modules')
    face_cropper = FaceDetector(model_dir='/store/joelee217/GazeTorch/cv/modules')

# Main: 
def main():
    req_lmks = False

    IMG_224_DIR.mkdir(exist_ok=True)
    IMG_NPY_DIR.mkdir(exist_ok=True)

    fns = [fn for fn in IMG_RAW_DIR.glob('*.png')]
    num_imgs = len(fns)
    print(f'Processing {num_imgs} images from {IMG_RAW_DIR} -> {IMG_224_DIR}')
    
    ids = [None] * num_imgs
    lmks = [None] * num_imgs
    hps = [None] * num_imgs

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=init_worker) as executor:
        
        chunk_size = 64
        
        results = list(tqdm(
            executor.map(process_image, fns, chunksize=chunk_size), 
            total=len(fns)
        ))

        for idx, (img_id, lmk, hp) in enumerate(results):
            ids[idx] = img_id
            lmks[idx] = lmk
            hps[idx] = hp

    np.save(IMG_NPY_DIR / 'gt_gazezero_vanilla.npy', 
            {'fn': np.array(ids),
             'headpose': np.array(hps)})
    
    print(np.array(ids))
    print(np.array(lmks))
    print(np.array(hps))
    print({'fn': np.array(ids, dtype=object),
             'headpose': np.array(hps)})

    if req_lmks:
        np.save(IMG_NPY_DIR / 'lmk_224.npy',
            np.array(lmks))

if __name__ == '__main__':
    main()