import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class GradientAlignment():
    '''
    aligned_losses_list: list of losses that we desire to align with the main loss, for example ['loss1', 'loss2'] in this case we may want to 
    align both segmentation and SCL gradeints to the main loss (probably gaze loss)
    main_loss_name: the name of loss which we want to align the gradients with it (for example: 'loss3')
    '''
    def __init__(self, list_aligned_losses, main_loss_name, model):
        self.list_aligned_losses = list_aligned_losses
        self.main_loss_name = main_loss_name
        self.model = model
        
    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad
    
    def _align_two_grads(self, main_grad, aux_grad, model):
        '''
        this function reutrns the aligned version of aux_grad on the main_grad
        '''
        # find common gradients between two losses
        common_main_grad = []
        common_aux_grad = []
        for i, (name, param) in enumerate(model.named_parameters()):
            if (main_grad[i] is not None) and (aux_grad[i] is not None):
                common_main_grad.append(main_grad[i])
                common_aux_grad.append(aux_grad[i])
        
        flatten_common_main_grad = self._flatten_grad(common_main_grad)
        flatten_common_aux_grad = self._flatten_grad(common_aux_grad)

        aligned_flatten_common_aux_grad = torch.zeros_like(flatten_common_aux_grad)
        dot_product = torch.dot(flatten_common_main_grad, flatten_common_aux_grad)
        if dot_product > 0:
            # no change is required
            return aux_grad
        else:
            # subtract the conflicting part from the auxiliary gradient vector
            projection_aux_grad_on_main_grad = (dot_product/torch.square(torch.norm(flatten_common_main_grad))) * flatten_common_main_grad
            aligned_flatten_common_aux_grad = flatten_common_aux_grad - projection_aux_grad_on_main_grad

        aligned_aux_grad = list(aux_grad)
        current_idx = 0
        # pack the gradients in the original form 
        for i, (name, param) in enumerate(model.named_parameters()):
            if (main_grad[i] is not None) and (aux_grad[i] is not None):
                shape = aux_grad[i].shape
                length = np.prod(shape)
                aligned_aux_grad[i] = aligned_flatten_common_aux_grad[current_idx:current_idx + length].view(shape).clone()
                current_idx += length
        
        return aligned_aux_grad
    
    def get_aligned_grads(self, dict_loss_grads):
        #   dict_loss_grads: the dicrionary which contains the gradients of the network and the keys are losses: {'loss1': grad_loss_1, ...}
        dict_aligned_losses_grads = dict()
        for loss_name in self.list_aligned_losses:
            aligned_aux_grad = self._align_two_grads(dict_loss_grads[self.main_loss_name], dict_loss_grads[loss_name], self.model)
            dict_aligned_losses_grads[loss_name] = aligned_aux_grad

        return dict_aligned_losses_grads
