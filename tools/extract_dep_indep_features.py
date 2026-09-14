import cv2
import torch
import numpy as np
from tqdm import tqdm
from model.net import Net_loader
from torch.utils.data import DataLoader
from tools.checkpoint import Checkpoint_io
from data.dataset_loader import get_dataset
from config.config_loader.config_loader import Config_loader, load_yaml

# base config file
CFG_PATH = '/data/panda/sma_ssl/lzh_data/GazeTorch/config/config_lib/sample_resnet_50_seg_ssl_xgaze_sma_baseline_gaze_infer.yaml'
cfg = Config_loader(CFG_PATH)
# ckpt dirs to extract features from dependent and independent modules
CKPT_DIRS = ['/data/panda/sma_ssl/Nov22/resnet50_face224_ssl_sma_baseline_gaze_0.0']
# batch size for inference
BATCH_SIZE = 16

for ckpt_dir in CKPT_DIRS:
    # load model
    cfg.net.ckpt.dir = ckpt_dir
    net_loader = Net_loader(cfg.net)
    model = net_loader.model
    model.eval()
    device = torch.device("cuda")
    model.to(device)
    dataset = get_dataset(cfg.data.dataset_val)
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=False)

    subjects = []
    person_dep_features = []
    person_indep_features = [] 
    iter = 0

    with torch.no_grad():
        for (X, Y_gt) in tqdm(dataloader):
            X = [x.cuda() for x in X]
            Y_gt = [y.cuda() for y in Y_gt]
            subject_cid = Y_gt[-1].cpu().detach().numpy()
            subjects.extend(subject_cid)
            y = model(X)
            output_ssl = y['head_ssl']
            # person dependent and independent features are outputs of SSL module
            fh1_perosn_dep, fh2_perosn_indep = output_ssl[0], output_ssl[1]
            fh1_perosn_dep = fh1_perosn_dep.cpu().detach().numpy()
            fh2_perosn_indep = fh2_perosn_indep.cpu().detach().numpy()
            person_dep_features.extend(fh1_perosn_dep)
            person_indep_features.extend(fh2_perosn_indep)
            iter += 1

    # save the features and subjects as numpy arrays
    person_indep_features = np.array(person_indep_features)
    person_dep_features = np.array(person_dep_features)
    subjects = np.array(subjects)
    np.save('p_dep_baseline_gaze_{}'.format(ckpt_dir.split('/')[-1]), person_dep_features)
    np.save('p_indep_baseline_gaze_{}'.format(ckpt_dir.split('/')[-1]), person_indep_features)
    np.save('subjects_baseline_gaze_{}'.format(ckpt_dir.split('/')[-1]), subjects)






