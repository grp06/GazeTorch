'''
Tester: evaluate a model on a target domain
by Medhi.
'''
import numpy as np
import torch

import logging
from tqdm import tqdm

from data.dataset_loader import get_dataset
from torch.utils.data import DataLoader
from config.config_loader.config_loader import load_yaml
from tools.param_constant import *
import tensorflow as tf


class Tester_tflite:   
    def __init__(self, tflite_model_path, cfg_dataset, classic_normalization=False, path_target=None):
        '''
What's required in the Config:
* tflite model path
Apart from Config, Tester takes 1 more arg:
* cfg_dataset: CFG loaded from a target dataset config file (config/eval/*.yaml)
* path_target (optional): if not None, store the gaze results into path_target, which should point to an NPY file.

Nov 7 '22 update
* classic_normalization
some old tflite models require mean and std values for normalization. 
if True, will pass mean=(0.485, 0.456, 0.406); std=(0.229, 0.224, 0.225).
        '''
        if isinstance(cfg_dataset, str):
            self.cfg_dataset = load_yaml(cfg_dataset)
        else:
            self.cfg_dataset = cfg_dataset

        self.cfg_dataset.args.img_channel_chw = False

        self.tflite_model_path = tflite_model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
        self.interpreter.allocate_tensors()

        # Create Dataset instance
        # batch_size=1 always
        self.cfg_data = cfg_dataset

        # configure mean & std
        if classic_normalization:
            self.cfg_data.args.augmenter_options = {
                'mean': np.array((0.485, 0.456, 0.406)),
                'std': np.array((0.229, 0.224, 0.225)),
            }

        self.dataset = get_dataset(self.cfg_data)
        self.dataloader = DataLoader(self.dataset)
        self.n = len(self.dataset)
        self.logger = logging.getLogger(__name__)
        self.logger.info("[Tester]Initialization done.")
    
        self.enable_w_wrt_subject_id = self.cfg_dataset.args.get("enable_w_wrt_subject_id", False)

        self.path_target = path_target

    def run(self):
        self.logger.info("[Tester]Begin eval.......")
        if self.path_target is not None:
            result = []

        from train.loss.angular import Angular_loss
        angular_loss_instance = Angular_loss()
        l1_loss_instance = torch.nn.L1Loss()
        sum_loss_angular = 0
        sum_loss_pitch = 0
        sum_loss_yaw = 0
        sum_w = 0
        for iter, (X, Y_gt) in tqdm(enumerate(self.dataloader)):
            # weight (w_wrt_subject_id) is always the last value
            if self.enable_w_wrt_subject_id:
                w = Y_gt[-1].numpy().squeeze()
            else:
                w = 1

            X = [x.float() for x in X]

            # Get input and output tensors.
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            tf_inp = X[0].cpu().detach().numpy()
            input_data = np.array(tf_inp, dtype=np.float32)
            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            self.interpreter.invoke()
            predicted_gaze = self.interpreter.get_tensor(output_details[0]['index'])[0].reshape([1, 2])
            true_gaze = Y_gt[0]

            if self.path_target is not None:
                result.append(predicted_gaze)

            predicted_gaze = torch.from_numpy(predicted_gaze)
            loss_ang = angular_loss_instance(predicted_gaze, true_gaze)
            loss_pitch = l1_loss_instance(predicted_gaze[:, 0], true_gaze[:, 0])
            loss_yaw = l1_loss_instance(predicted_gaze[:, 1], true_gaze[:, 1])
            sum_loss_angular += loss_ang * w
            sum_loss_pitch += loss_pitch * w
            sum_loss_yaw += loss_yaw * w

            sum_w += w
        
        print("[Tester]Voila. Ang loss = {}, Pitch loss = {}, Yaw loss= {}".format(sum_loss_angular / sum_w, sum_loss_pitch / sum_w, sum_loss_yaw / sum_w))
        if self.path_target is not None:
            np.save(self.path_target, np.array(result).squeeze())
            print("Results saved to:", self.path_target)
        
        return [sum_loss_angular / sum_w, sum_loss_pitch / sum_w, sum_loss_yaw / sum_w]
