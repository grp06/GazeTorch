import os, cv2, sys
import numpy as np
sys.path.append('/store/joelee217/GazeTorch/')
from cv.headpose import HeadposeDetector
from tqdm import tqdm
import concurrent.futures
from app.gaze import HeadPoseEstimator

GC3D_ROOT = '/data/gazecapture_3d'
GC3D_INVALID_ID = 'invalid_ids_v2.npy'
GC3D_GT_DEFAULT = 'gt_p17y17_with_conf_v3.npy'

GAZEGENE_ROOT = '/data/GazeGene'
GAZEGENE_ID = 'valid_ids_roll_pi36.npy'
GAZEGENE_GT = 'gazegene_gt_conf_v2.npy' 

XGAZE_ROOT = '/data/xgaze/448'
XGAZE_GT = 'gt_p17y17b16_with_confs_v3.npy'


# # Load GC3D and invalid ids
# gc3d_gt = np.load(os.path.join(GC3D_ROOT, GC3D_GT_DEFAULT), allow_pickle=True).item()

# # >>> (['fn', 'gaze', 'gaze_L', 'gaze_R', 'num_data', 'pitch_cid', 
# # 'yaw_cid', 'start_ids', 'pitch_nbins', 'yaw_nbins', 'pitch_bias', 
# # 'bin_size', 'subject_cid', 'w_L', 'w_R', 'headpose',
# #  'wE_L', 'wI_L', 'wE_R', 'wI_R', 'w_wrt_subject_id'])

# invalid_ids = np.load(os.path.join(GC3D_ROOT, GC3D_INVALID_ID))
# gc3d_gt_tf_headpose = gc3d_gt

# gazegene_gt = np.load(os.path.join(GAZEGENE_ROOT, GAZEGENE_GT), allow_pickle=True).item()
# gazegene_valid_ids = np.load(os.path.join(GAZEGENE_ROOT, GAZEGENE_ID))
# gazegene_gt_tf_headpose = gazegene_gt

xgaze_gt = np.load(os.path.join(XGAZE_ROOT, XGAZE_GT), allow_pickle=True).item()
xgaze_gt_tf_headpose = xgaze_gt
GENERIC_FACE_MODEL_PATH = "tools/alex/face_generic.npy"
generic_face_model = np.load(GENERIC_FACE_MODEL_PATH)


# need to initialize tf_lite model for each process
def init_worker():
    global poser
    # fd_model = HeadposeDetector(model_dir='/data/joelee217/GazeTorch/cv/modules')
    poser = HeadPoseEstimator(generic_face_model, face_landmark_ids=(0,1,2,3,4,5))

def gc3d_process_tf_headpose(args):
    idx, file_path, root_dir = args 

    full_path = os.path.join(root_dir, 'face_224', file_path)
    img = cv2.imread(full_path)
    
    if img is None:
        return (idx, None) 

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    try:
        img_face, R_norm, l2d = poser.run(img, do_normalization=True)
        hR, _ = cv2.Rodrigues(poser.rvec)
        head_mat = R_norm * hR  # from GazeTorch/data/gazecapture/generate_gt.py
        subject_headpose = np.array([np.arcsin(head_mat[1, 2]),
                        np.arctan2(head_mat[0, 2], head_mat[2, 2])])

    except Exception:
        return (idx, None)

    if subject_headpose is None:
        return (idx, None)

    res = np.array([np.degrees(subject_headpose[0]), np.degrees(subject_headpose[1]), 0])
    return (idx, res)

if __name__ == "__main__":
    
    face224 = os.listdir(os.path.join(XGAZE_ROOT, 'face_224'))
    num_files = len(face224)

    if 'mp_headpose' not in xgaze_gt_tf_headpose:
        xgaze_gt_tf_headpose['mp_headpose'] = np.zeros((num_files, 1, 3))
    
    task_args = []
    for idx, f in enumerate(face224):
        task_args.append((idx, f, XGAZE_ROOT))

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=init_worker) as executor:
        
        chunk_size = 50
        
        results = list(tqdm(
            executor.map(gc3d_process_tf_headpose, task_args, chunksize=chunk_size), 
            total=len(task_args)
        ))

        for idx, tf_headpose in results:
            if tf_headpose is not None:
                xgaze_gt_tf_headpose['mp_headpose'][idx] = tf_headpose

    # Save 
    save_dir = os.path.join(XGAZE_ROOT, 'xgaze_gt_mp_headpose.npy')
    np.save(save_dir, xgaze_gt_tf_headpose)
    print(f"Saved at {save_dir}")
