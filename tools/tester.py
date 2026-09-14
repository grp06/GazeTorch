'''
Tester: evaluate a model on a target domain
by Zhenhao, initial version Sep 22 2022
'''
import numpy as np
import os
import torch
import cv2

import logging
from tqdm import tqdm

from data.dataset_loader import get_dataset
from torch.utils.data import DataLoader
from config.config_loader.config_base import Config_base
from config.config_loader.config_loader import load_yaml
from model.net import Net_loader
from train.loss_loader import get_loss
from tools.param_constant import *
from app.gaze import Gaze3D_pytorch

class GazeEvaluator:
    def __init__(self) -> None:
        from train.loss.angular import Angular_loss
        self.module_l1 = torch.nn.L1Loss()
        self.module_ang = Angular_loss()

    def __call__(self, t1:torch.Tensor, t2:torch.Tensor):
        loss_ang = self.module_ang(t1, t2)
        loss_pitch = self.module_l1(t1[...,0], t2[...,0])
        loss_yaw = self.module_l1(t1[...,1], t2[...,1])
        return float(loss_ang), float(loss_pitch), float(loss_yaw)

class Tester:   
    def __init__(self, cfg:Config_base, cfg_dataset, path_target=None,
        device=0,
        result_module:str=None, result_tensor_id:int=0,
        gt_tensor_id:int=0):
        '''
        What's required in the Config:
        * net
            * network architecture
        * train
            * ckpt
            * loss_val function 

        Apart from Config, Tester takes:
        * cfg_dataset: CFG loaded from a target dataset config file (config/eval/*.yaml)
        * path_target (optional): if not None, store the gaze results into path_target, which should point to an NPY file.
        * device
            * if "cpu", then use CPU
            * if an int K, then use the GPU #K

        * result_module, result_tensor_id: the model output module & tensor ID (in cfg.net.output)
            * if result_module=None, then just apply the first output module.
            * Feb 6 '23: will fully override `cfg.net.output`.
            * This may disrupt loss & evaluation. For example if user makes it an intermediate feature layer, then `--skip_eval` is recommended.
        * gt_tensor_id: the GT tensor ID (in dataset.__getitem__)
        '''
        if isinstance(cfg_dataset, str):
            self.cfg_dataset = load_yaml(cfg_dataset)
        else:
            self.cfg_dataset = cfg_dataset
        self.cfg = cfg

        self.gpu = device!='cpu'
        if device=='cpu':
            self.device = torch.device('cpu')
        else:
            try:
                device = int(device)
            except:
                device = 0
            self.device = torch.device(f'cuda:{device}')

        self.result_module = result_module
        self.result_tensor_id = result_tensor_id
        self.gt_tensor_id = gt_tensor_id
        if result_module and result_module not in cfg.net.output:
            cfg.net.output = [result_module] # WARNING: this may disrupt loss calculation
        
        # Construct the Network & Load ckpt
        self.net_loader = Net_loader(cfg.net)
        self.start_epoch = self.net_loader.start_epoch
        self.model = self.net_loader.model
        self.model = self.model.to(self.device)

        # Create Dataset instance
        # batch_size=1 always
        self.cfg_data = cfg_dataset
        self.dataset = get_dataset(self.cfg_data)
        self.dataloader = DataLoader(self.dataset)
        self.n = len(self.dataset)

        # Construct loss func (legacy)
        # self.loss_cfg, self.loss_modules, self.loss_fns = self.initialize_loss()
        self.eval = GazeEvaluator()

        self.enable_w_wrt_subject_id = self.cfg_dataset.args.get("enable_w_wrt_subject_id", False)

        self.path_target = path_target
        if self.path_target is not None and not os.path.isdir(self.path_target):
            os.mkdir(self.path_target)

        self.logger = logging.getLogger(__name__)
        self.logger.info("[Tester]Initialization done.")

    @torch.no_grad()
    def initialize_loss(self):
        ## we first construct all loss mentioned, and used in this net config
        loss_cfg = self.cfg.train.loss
        loss_modules = [(name, config) for name, config in loss_cfg.__dict__.items() if isinstance(config, Config_base)]
        loss_fns = {}
        for name, config in loss_modules:
            loss_type = config.type
            loss_args = config.get("args", {})
            loss_fns[name] = get_loss(loss_type, **loss_args)
        return loss_cfg, loss_modules, loss_fns
    
    def run(self, skip_eval=False):
        '''
        skip_eval: if True, don't compute loss, just take the output (and save.)
        '''
        self.logger.info("[Tester]Begin eval.......")
        self.model.eval()
        if self.path_target is not None:
            results = []

        losses = []
        sum_w = 0
        for iter, (X, Y_gt) in tqdm(enumerate(self.dataloader)):
            X = [x.float() for x in X]
            gt = Y_gt[self.gt_tensor_id].squeeze() # the second [0] gets 1st item from MB; 
            if self.gpu:
                X = [x.to(self.device) for x in X]
                if isinstance(gt, np.ndarray):
                    gt = torch.from_numpy(gt)
                gt = gt.to(self.device)

            # weight (w_wrt_subject_id) is always the last value
            if self.enable_w_wrt_subject_id:
                w = Y_gt[-1].numpy().squeeze()
            else:
                w = 1
            
            Y_pred = self.model(X)
            result = Y_pred[self.result_module][self.result_tensor_id] # BCHW

            if not skip_eval:
                # legacy
                # _, loss_cpu = self.forward_loss(Y_pred, Y_gt, self.loss_flag)
                loss_vec = np.array(self.eval(result, gt)) * w
                losses.append(loss_vec)
                sum_w += w

            if self.path_target is not None:
                result_np = result.cpu().detach().numpy()
                results.append(result_np)
        
        if not skip_eval:
            ans = np.sum(losses, axis=0) / sum_w
            print(f"[Tester]Voila.\nAngular loss = {ans[0]}\nPitch L1 loss = {ans[1] * 180 / np.pi}\nYaw L1 loss = {ans[2] * 180 / np.pi}\n")
            return ans
        if self.path_target is not None:
            np.save(self.path_target, np.array(results))
            print("[Tester]Results saved to:", self.path_target)

    def demo_segmentation(self, step=1):
        if self.path_target is None:
            print("[Tester]No target dir!")
            return
        if not os.path.exists(self.path_target):
            os.makedirs(self.path_target)
        from tools.utils import visualize_facemap
        self.logger.info("[Tester]Begin eval.......")
        self.model.eval()

        for i in range(0, len(self.dataset), step):
        # for iter, (X, Y_gt) in tqdm(enumerate(self.dataloader)):
            # if not iter % step == 0:
            #     continue
            X, Y_gt = self.dataset[i]
            face = X[0]
            if self.dataset.chw:
                # face = face.numpy()
                face = face.transpose((1,2,0)) # CHW=>HWC
                face = cv2.normalize(face, None, 0, 1, cv2.NORM_MINMAX)
            X[0] = X[0][None, ...]

            X = [torch.from_numpy(x) for x in X]
            gt = Y_gt[self.gt_tensor_id].squeeze() # the second [0] gets 1st item from MB; 
            if self.gpu:
                X = [x.to(self.device) for x in X]
                if isinstance(gt, np.ndarray):
                    gt = torch.from_numpy(gt)
                gt = gt.to(self.device)
            
            Y_pred = self.model(X)
            facemap = Y_pred[self.result_module][self.result_tensor_id][0]
            facemap = facemap.cpu().detach()
            rgb = visualize_facemap(facemap, face)

            path = os.path.join(self.path_target, f"{i}.png")
            rgb = rgb[:,:,::-1] # RGB=>BGR
            cv2.imwrite(path, rgb)

        print("[Tester]Results saved to:", self.path_target)

    # legacy - Tester doesn't need to get loss tensors in training
    # def forward_loss(self, Ys_pred, Ys_gt, loss_flag='val'):
    #     # We first try to calculate all the loss used in loss composistion,
    #     # then store the result under losses_dict
    #     losses_dict = {}

    #     # loss_form stores the list of loss terms with cfg.
    #     # Some loss terms not included in loss_train/val can be skipped
    #     loss_form = None
    #     loss_key = f"loss_{loss_flag}"
    #     loss_form = self.loss_cfg.get(loss_key, None)
    #     if loss_form is None:
    #         raise Exception("[Tester]ERROR: Invalid loss flag {}.".format(loss_flag))
        
    #     for name, config in self.loss_modules:
    #         if name not in loss_form:
    #             continue
    #         loss_func = self.loss_fns[name]
    #         loss_input = []
    #         # in this step we are preparing the all the input that will be feed into the loss module
    #         # note that due to the nature of Pytroch, it better to cast everything to a float
    #         # Notes for Tester: 
    #         # DataLoader is CPU-based, so data has to be manually transferred to GPU (Line 86, 88)
    #         for source, split in config.module_input.items():
    #             if source == DATASET:
    #                 loss_input.append(Ys_gt[split].float().to(self.device))
    #             else:
    #                 if split == ALL:
    #                     for pred in Ys_pred[source]:
    #                         loss_input.append(pred.float().to(self.device))
    #                 else:
    #                     loss_input.append(Ys_pred[source][split].float().cuda(self.device))
    #         # feed arguments (*loss_input) and invoke the loss function.
    #         # as a result we have the loss_batch
    #         loss_batch = loss_func(*loss_input)
    #         losses_dict[name] = loss_batch

    #     # Total loss will record the addition result of all the loss together
    #     total_loss = None
    
    #     for key, value in loss_form.items():
    #         if total_loss is None:
    #             total_loss = torch.mul(losses_dict[key], value)
    #         else:
    #             total_loss += torch.mul(losses_dict[key], value)
    #     return total_loss, total_loss.cpu().detach().numpy()
        

class Tester_pkl:   
    def __init__(self, model_path, cfg_dataset, path_target=None,
        device=0,
        result_module:str=None, result_tensor_id:int=0,
        gt_tensor_id:int=0):
        '''
        What's required in the Config:
        * net
            * network architecture
        * train
            * ckpt
            * loss_val function 

        Apart from Config, Tester takes:
        * cfg_dataset: CFG loaded from a target dataset config file (config/eval/*.yaml)
        * path_target (optional): if not None, store the gaze results into path_target, which should point to an NPY file.
        * device
            * if "cpu", then use CPU
            * if an int K, then use the GPU #K

        * result_module, result_tensor_id: the model output module & tensor ID (in cfg.net.output)
            * if result_module=None, then just apply the first output module.
            * Feb 6 '23: will fully override `cfg.net.output`.
            * This may disrupt loss & evaluation. For example if user makes it an intermediate feature layer, then `--skip_eval` is recommended.
        * gt_tensor_id: the GT tensor ID (in dataset.__getitem__)
        '''
        if isinstance(cfg_dataset, str):
            self.cfg_dataset = load_yaml(cfg_dataset)
        else:
            self.cfg_dataset = cfg_dataset

        self.gpu = device!='cpu'
        if device=='cpu':
            self.device = torch.device('cpu')
        else:
            try:
                device = int(device)
            except:
                device = 0
            self.device = torch.device(f'cuda:{device}')

        self.result_module = result_module
        self.result_tensor_id = result_tensor_id
        self.gt_tensor_id = gt_tensor_id
        
        # Construct the Network & Load ckpt
        self.model = Gaze3D_pytorch(model_path)

        # Create Dataset instance
        # batch_size=1 always
        self.cfg_data = self.cfg_dataset
        self.dataset = get_dataset(self.cfg_data)
        self.dataloader = DataLoader(self.dataset)
        self.n = len(self.dataset)

        # Construct loss func (legacy)
        # self.loss_cfg, self.loss_modules, self.loss_fns = self.initialize_loss()
        self.eval = GazeEvaluator()

        self.enable_w_wrt_subject_id = self.cfg_dataset.args.get("enable_w_wrt_subject_id", False)

        self.path_target = path_target
        if self.path_target is not None and not os.path.isdir(self.path_target):
            os.mkdir(self.path_target)

        self.logger = logging.getLogger(__name__)
        self.logger.info("[Tester]Initialization done.")

    @torch.no_grad()
    def initialize_loss(self):
        ## we first construct all loss mentioned, and used in this net config
        loss_cfg = self.cfg.train.loss
        loss_modules = [(name, config) for name, config in loss_cfg.__dict__.items() if isinstance(config, Config_base)]
        loss_fns = {}
        for name, config in loss_modules:
            loss_type = config.type
            loss_args = config.get("args", {})
            loss_fns[name] = get_loss(loss_type, **loss_args)
        return loss_cfg, loss_modules, loss_fns
    
    def run(self, skip_eval=False):
        '''
        skip_eval: if True, don't compute loss, just take the output (and save.)
        '''
        self.logger.info("[Tester]Begin eval.......")
        # self.model.eval()
        if self.path_target is not None:
            results = []

        losses = []
        sum_w = 0
        for iter, (X, Y_gt) in tqdm(enumerate(self.dataloader)):
            X = [x.float() for x in X]
            gt = Y_gt[self.gt_tensor_id].squeeze() # the second [0] gets 1st item from MB; 
            if self.gpu:
                X = [x.to(self.device) for x in X]
                if isinstance(gt, np.ndarray):
                    gt = torch.from_numpy(gt)
                gt = gt.to(self.device)

            # weight (w_wrt_subject_id) is always the last value
            if self.enable_w_wrt_subject_id:
                w = Y_gt[-1].numpy().squeeze()
            else:
                w = 1
            
            Y_pred = self.model.cnn(X)
            result = Y_pred[self.result_module][self.result_tensor_id] # BCHW

            # # print results
            if iter % 1000 == 0:
                result_cpu = result.detach().cpu().numpy() * 180 / np.pi
                gt_cpu = gt.detach().cpu().numpy() * 180 / np.pi
                print(f"iter:{iter} GT:({gt_cpu[0]:.3f}, {gt_cpu[1]:.3f}); Pred:({result_cpu[0,0]:.3f}, {result_cpu[0,1]:.3f})")
            # if abs(result_cpu[0,1] - gt_cpu[1]) > 0.2:
            #     print(f"ID #{iter}: issue YAW")
            #     print(f"GT:({gt_cpu[0]}, {gt_cpu[1]}); Pred:({result_cpu[0,0]}, {result_cpu[0,1]})")
            # # if abs(result_cpu[0,0] - gt_cpu[0]) > 0.2:
            # #     print("ID #i: issue PITCH")
            # #     print(f"GT:({gt_cpu[0]}, {gt_cpu[1]}); Pred:({result_cpu[0,0]}, {result_cpu[0,1]})")

            if not skip_eval:
                # legacy
                # _, loss_cpu = self.forward_loss(Y_pred, Y_gt, self.loss_flag)
                loss_vec = np.array(self.eval(result, gt)) * w
                losses.append(loss_vec)
                sum_w += w

            if self.path_target is not None:
                result_np = result.cpu().detach().numpy()
                results.append(result_np)
        
        if not skip_eval:
            ans = np.sum(losses, axis=0) / sum_w
            print(f"[Tester]Voila.\nAngular loss = {ans[0]}\nPitch L1 loss = {ans[1] * 180 / np.pi}\nYaw L1 loss = {ans[2] * 180 / np.pi}\n")
            return ans
        if self.path_target is not None:
            np.save(self.path_target, np.array(results))
            print("[Tester]Results saved to:", self.path_target)