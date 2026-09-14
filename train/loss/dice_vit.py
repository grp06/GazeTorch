import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn


def CalculateDiceLoss(input, target):

    eps = 1e-6

    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))
    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                            .format(input.shape))
    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {}"
                            .format(input.shape, input.shape))
    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {}" .format(
                input.device, target.device))
    
    # compute sigmoid over the classes axis

    # Zheng(25-12-05): input is Nx1xHxW, use F.sigmoid
    input_sig = F.sigmoid(input)

    # create the labels one hot tensor
    # target_one_hot = one_hot(target, num_classes=input.shape[1],
    #                          device=input.device, dtype=input.dtype)
    
    target_one_hot = torch.eye(input.shape[1], device="cuda")[target.squeeze(1)]
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).cuda().float()
    target_one_hot = target_one_hot[:, 0, :, :].unsqueeze(1)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_sig * target_one_hot, dims)
    cardinality = torch.sum(input_sig + target_one_hot, dims)

    dice_score = 2. * intersection / (cardinality + eps)
    # print("Loss: ", (1. - dice_score).shape)

    return (1. - dice_score)

class DiceLoss(_Loss):
    def __init__(self, eyelid_mode='merge') -> None:
        '''
        eyelid_mode:
        'eyelid': treat eyelid as an independent channel. Seg head outputs 3 results: eye + eyelid + iris - 3-classification.
        'merge': treat eyelid as part of eye. Seg head outputs 2 results.
            * need processing for "v3" GT segmaps' eye channel (0 = none, 1 = eye, 2 = eyelid): facemap[facemap>1] = 1
        'ignore': ignore eyelid.
            * need processing for "v3" GT segmaps' eye channel (0 = none, 1 = eye, 2 = eyelid): facemap[facemap>1] = 0
        '''
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6
        self.eyelid = eyelid_mode
        self.c_iris = 1 # align with old method, just specify number of channel to 1
        if self.eyelid=='eyelid':
            self.c_eye = 2
        else:
            self.c_eye = 1

    def _remove_eyelid_labels(self, facemap, mode):
        if mode == 'merge':
            facemap[facemap>1] = 1
        elif mode == 'ignore':
            facemap[facemap>1] = 0
        return facemap

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        '''
        input should be B*4*H*W
        output should be [5] Tensors:
            * 0: B*4*H*W, the ground truth segmentation result
            * 1-4: all scalar - 4 weights (Line 99-104)
        '''
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        # if not input.device == target.device:
        #     raise ValueError(
        #         "input and target must be in the same device. Got: {}" .format(
        #             input.device, target.device))

        segmap = target[0]
        if self.eyelid != 'eyelid':
            segmap = self._remove_eyelid_labels(segmap, self.eyelid)

        # Confidence score for eyes and iris maps
        weight_l_eye = torch.where(target[1] >= 0.5, 1, 0).cuda()
        weight_r_eye = torch.where(target[3] >= 0.5, 1, 0).cuda()

        weight_l_iris = torch.where(target[2] >= 0.5, 1, 0).cuda()
        weight_r_iris = torch.where(target[4] >= 0.5, 1, 0).cuda()

        c = 0
        loss_eye_L = CalculateDiceLoss(input[:, c:c+self.c_eye, :, :], segmap[:, 0, :, :].unsqueeze(1).long())
        loss_eye_L = torch.mean(loss_eye_L * weight_l_eye.squeeze())
        
        c += self.c_eye
        loss_iris_L = CalculateDiceLoss(input[:, c:c+self.c_iris, :, :], segmap[:, 1, :, :].unsqueeze(1).long())
        loss_iris_L = torch.mean(loss_iris_L * weight_l_iris.squeeze())

        c += self.c_iris
        loss_eye_R = CalculateDiceLoss(input[:, c:c+self.c_eye, :, :], segmap[:, 2, :, :].unsqueeze(1).long())
        loss_eye_R = torch.mean(loss_eye_R * weight_r_eye.squeeze())

        c += self.c_eye
        loss_iris_R = CalculateDiceLoss(input[:, c:c+self.c_iris, :, :], segmap[:, 3, :, :].unsqueeze(1).long())
        loss_iris_R = torch.mean(loss_iris_R * weight_r_iris.squeeze())

        loss_seg = loss_eye_L + loss_iris_L + loss_eye_R + loss_iris_R
    

        return loss_seg


if __name__ == '__main__':
    input_tensor = torch.rand(2, 4, 224, 224)
    target_tensor = [torch.rand(2, 4, 224, 224).cuda().float(), torch.rand(2, 1), torch.rand(2, 1), torch.rand(2, 1), torch.rand(2, 1)]
    loss = DiceLoss()
    output = loss(input_tensor.cuda().float(), target_tensor)
    print(input_tensor, target_tensor)
    print(output)
