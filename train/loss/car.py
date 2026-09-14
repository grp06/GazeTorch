import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from torch.nn.modules.loss import _Loss

from .angular import Angular_loss
from tools.torch_ops import angle_transform

DEFAULT_THRESHOLD = 8 * np.pi / 180

class Car_loss(_Loss):
    '''
    Car Loss.
    '''
    def __init__(self, nbins_pitch=17, nbins_yaw=17, bin_size=4., 
                beta_p=1.0, beta_y=1.0, lambda_pitch=1., lambda_yaw=1.,
                pitch_bias=0.0, temperature=1., angular_error=False, class_weight=False,
                ordinal=False, var=False, lambda_var=0.5):
        super(Car_loss, self).__init__()
        
        self.nbins_pitch = nbins_pitch
        self.nbins_yaw = nbins_yaw
        self.bin_size = bin_size
        self.beta_p = beta_p
        self.beta_y = beta_y
        self.lambda_pitch = lambda_pitch
        self.lambda_yaw = lambda_yaw
        self.pitch_bias = pitch_bias

        self.angular_error = angular_error
        self.class_weight = class_weight

        self.temperature = temperature

        self.ordinal = ordinal

        self.var = var
        self.lambda_var = lambda_var

        if class_weight:
            # class weight here may be changed in the future..
            if pitch_bias == -12.0: # class weight for Hybrid dataset
                self.pitch_class_weight = torch.tensor([0.43458254, 2.71611006, 2.13711785, 1.62758827, 1.40612248,
                                                1.07788212, 0.76116293, 0.60192933, 0.50132412, 0.27910121,
                                                0.24672503, 0.39525608, 1.0151933 , 1.54698465, 0.25292004])
                
                # yaw_class_weight = tf.constant([0.29494198, 2.18313782, 1.7864425 , 1.25627772, 0.86873456,
                #                                 0.69802294, 0.659883  , 0.44797907, 0.37876704, 0.35449272,
                #                                 0.58396691, 0.71135047, 0.89737461, 1.27790613, 1.88650424,
                #                                 2.37466942, 0.33954886]) # old yaw weight for Hybrid

                self.yaw_class_weight = torch.tensor([1.0, 2.00098051, 1.63738477, 1.15145604, 0.79624882, 0.63978109, 0.60482348,
                                                0.41060045, 0.34716336, 0.32491445, 0.53524170, 0.65199659, 0.82249919,
                                                1.17127981, 1.72909753, 2.17653104, 1.0]) # Zheng(23-04-05): new yaw weight for Hybrid, bin 0&16 = 1.0, sum of rest = 15
            elif pitch_bias == -14.5:
                self.pitch_class_weight = torch.tensor([5.56870604, 5.56870604, 1.76224875, 0.81652581, 0.37779553,
                                                0.2297321 , 0.12309253, 0.07980376, 0.05583222, 0.05056943,
                                                0.04969397, 0.05133394, 0.05237684, 0.09742313, 0.11615991])
                self.yaw_class_weight = torch.tensor([2.9220589 , 1.74417469, 1.28478821, 0.99943825, 0.6771965 ,
                                                0.52401406, 0.50947871, 0.50947871, 0.51063497, 0.51531299,
                                                0.51384193, 0.51857917, 0.71005455, 1.00670486, 1.1756945 ,
                                                1.49500688, 1.38354211])
            else: # class weight for XGaze dataset
                self.pitch_class_weight = torch.tensor([2.11116927, 0.42382029, 0.44301028, 0.47399741, 0.57758744,
                                                0.75299389, 0.78926164, 0.7240516 , 0.69036799, 0.6892894 ,
                                                0.65857192, 0.63250588, 0.71180506, 0.8090134 , 0.79211542,
                                                0.73724192, 4.98319719])
                self.yaw_class_weight = torch.tensor([4.38756576, 0.55212811, 0.58055357, 0.59203611, 0.59210352,
                                                0.58680043, 0.59587861, 0.61720332, 0.67367224, 0.61349565,
                                                0.59938404, 0.57713802, 0.57037434, 0.56529595, 0.5323314 ,
                                                0.5037037 , 3.86033521])

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        '''
        :param input: Nx(nbins_pitch + nbins_yaw) tensor.
        :param target: list [gt_gaze, gt_bin]. Both shape as [Nx2]
            * when using dataset_id, it should be a list [gt_gaze, gt_bin, dataset_id(Nx1)]
        :return: car + gaze regression error.
        '''

        input_pitch = input[:, :self.nbins_pitch]
        input_yaw = input[:, self.nbins_pitch:self.nbins_pitch + self.nbins_yaw]

        if self.var:
            pass

        target_ce_pitch = target[1][:, 0]
        target_ce_yaw = target[1][:, 1]
        target_reg_pitch = target[0][:, 0]
        target_reg_yaw = target[0][:, 1]

        dataset_dependent = len(target) == 3 # here gt should be ['gaze', 'bin', 'dataset_id']
        if dataset_dependent:
            dataset_id = target[-1]
            dataset_mask = (dataset_id == 0).to(int) # LZH(25.11.10): block GC3D & GazeGene pitch labels

            input_pitch = input_pitch * (torch.unsqueeze(dataset_mask, 1).repeat(1, self.nbins_pitch))

            # if self.ordinal:
            pitch_label = target_reg_pitch.clone()

            target_ce_pitch = target_ce_pitch * dataset_mask
            target_reg_pitch = target_reg_pitch * dataset_mask

        # CE loss
        # add class weight from pre-calculated values
        if self.class_weight:
            loss_ce_pitch = F.cross_entropy(input_pitch, target_ce_pitch, weight=self.pitch_class_weight)
            loss_ce_yaw = F.cross_entropy(input_yaw, target_ce_yaw, weight=self.yaw_class_weight)
        else:
            loss_ce_pitch = F.cross_entropy(input_pitch, target_ce_pitch)
            loss_ce_yaw = F.cross_entropy(input_yaw, target_ce_yaw)

        # regression loss
        pitch_angle = angle_transform(input_pitch, self.nbins_pitch, self.bin_size, self.pitch_bias, self.temperature)
        yaw_angle = angle_transform(input_yaw, self.nbins_yaw, self.bin_size, self.temperature)

        if dataset_dependent:
            if self.ordinal:
                pitch_pred = pitch_angle.clone()

                temp_gt = pitch_label.clone().squeeze()
                temp_gt = temp_gt[0]
                temp_pred = pitch_pred.clone().squeeze()
                temp_pred = temp_pred[0]

                print("[train debug] pitch_gt={:.4f}; pred={:.4f}".format(temp_gt, temp_pred))

            # if ordinal, pitch_pred always around -0.2094

            pitch_angle = pitch_angle.squeeze() * dataset_mask

        loss_reg_pitch = F.l1_loss(pitch_angle.squeeze(), target_reg_pitch)
        loss_reg_yaw = F.l1_loss(yaw_angle.squeeze(), target_reg_yaw)

        loss_total_pitch = loss_ce_pitch + self.beta_p*loss_reg_pitch
        loss_total_yaw = loss_ce_yaw + self.beta_y*loss_reg_yaw

        loss_total = self.lambda_pitch*loss_total_pitch + self.lambda_yaw*loss_total_yaw

        if self.ordinal:
            loss_total += ordinal_loss(pitch_label, pitch_pred)

        # variance loss 
        if self.var:
            loss_total += angle_transform(input_pitch, self.nbins_pitch, self.bin_size, self.pitch_bias, var=True) * self.lambda_var
            loss_total += angle_transform(input_yaw, self.nbins_yaw, self.bin_size, var=True) * self.lambda_var

        # mainly for validation period, return angular error:
        if self.angular_error:
            gaze_input = torch.cat([pitch_angle, yaw_angle], dim=1)
            loss_ang = Angular_loss()
            angular_error = loss_ang(gaze_input, target[0])
            return angular_error

        return loss_total


def ordinal_loss(pitch_label, pitch_pred, threshold=DEFAULT_THRESHOLD):
    # pitch_lable should be B x 1
    pitch_label = pitch_label.view(-1, 1)
    pitch_pred = pitch_pred.view(-1, 1)
    M_matrix = (pitch_label - pitch_label.T) < -threshold # B x B
    M_matrix = M_matrix.to(int)
    D_matrix = pitch_pred - pitch_pred.T # why always 6931?
    D_matrix = D_matrix * M_matrix
    D_matrix.clamp_min_(0.)

    # l = torch.log(torch.exp(D_matrix) + 1)
    l = torch.sum(D_matrix)
    # print(l)
    return l