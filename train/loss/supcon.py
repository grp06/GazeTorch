import torch
import torch.nn as nn
import torch.nn.functional as F

'''
SupCon loss module.
adversarial:
    * if True, then treat non-equal-pair as positive pairs - for adversarial learning.
'''

def pitchid_datasetid_mask(pitch_ids, dataset_ids, nbins_pitch):
    n = pitch_ids.shape[0]
    
    mask = torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            b_equalP = pitch_ids[i] == pitch_ids[j]
            b_equalD = dataset_ids[i] == dataset_ids[j]
            b_extreme = pitch_ids[i] == 0 or pitch_ids[i] == nbins_pitch-1
            mask[i][j] = b_equalP and (b_equalD or b_extreme)
    
    return mask.to(int)

GLASSES_MASK = None
def glasses_mask(batch_size, adversarial=True):
    # returns a constant mask for 4-view-with-glasses setting (view 0,1 = no glases, view 2,3 = with glasses)
    n = batch_size
    mask = torch.zeros(n, n, dtype=int)

    if adversarial:
        mask[n//2:, :n//2] = 1
        mask[:n//2, n//2:] = 1
    else:    
        mask[:n//2, :n//2] = 1
        mask[n//2:, n//2:] = 1

    return mask

def compute_supcon_loss(features, labels, temperature=0.07, base_temperature=0.07,
                        adversarial=False,
                        dataset_ids=None, nbins_pitch=9,
                        n_views=2, glasses_mode=False, given_mask=None):
    features = torch.nn.functional.normalize(features, p=2.0, dim=1)

    if glasses_mode:
        n_views = 1
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1) # (n,1)
        if given_mask.shape[0] != batch_size:
            mask = glasses_mask(batch_size, adversarial=adversarial).cuda()
        else:
            mask = given_mask
    else:
        batch_size = features.shape[0] // n_views
        if not isinstance(labels, torch.Tensor):
            print('supcon:')
            print(labels)
            print(type(labels))
            labels = torch.tensor(labels).view(-1, 1).cuda()
        labels = labels[0:labels.shape[0] // n_views]
        labels = labels.contiguous().view(-1, 1) # (n,1)
        
        # get the mask
        if adversarial:
            mask = torch.ne(labels, labels.T).float() # (n,n)
        else:
            if dataset_ids is not None:
                mask = pitchid_datasetid_mask(labels, dataset_ids, nbins_pitch)
            else:
                mask = torch.eq(labels, labels.T).float()
        mask = mask.cuda()

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(features, features.T), temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask (dependent on view #)
    mask = mask.repeat(n_views, n_views)
    # mask-out self-contrast cases
    # print(torch.ones_like(mask), torch.arange(batch_size * n_views).view(-1, 1))
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * n_views).view(-1, 1).cuda(), 0)
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    # compute mean of log-likelihood over positive
    # guard: an anchor whose label is unique in the batch has NO positive pair
    # after self-contrast removal -> mask.sum(1)==0 -> 0/0 = NaN (in fwd AND in
    # DivBackward). Floor the denominator to 1 for those rows; their numerator
    # is already 0, so they contribute 0 loss instead of NaN.
    mask_pos_pairs = mask.sum(1)
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6,
                                 torch.ones_like(mask_pos_pairs),
                                 mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(n_views, batch_size).mean()

    return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf"""
    def __init__(self, temperature=0.07, base_temperature=0.07, adversarial=False,
                 n_views=2, glasses_mode=False, batch_size=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.adversarial = adversarial
        self.n_views = n_views
        self.glasses_mode = glasses_mode
        if self.glasses_mode:
            self.glasses_mask = glasses_mask(batch_size, adversarial).cuda()
        else:
            self.glasses_mask = None

    def forward(self, features, labels):
        '''
        features: a single Tensor.
        adversarial:
            * if False, this loss tries to make the features CORRELATED to labels.
            * if True, this loss tries to make the features INDEPENDENT of labels.
        '''
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze()
        neg_loss_supcon = torch.mean(
            compute_supcon_loss(features, labels, adversarial=self.adversarial,
                                n_views=self.n_views,
                                glasses_mode=self.glasses_mode, given_mask=self.glasses_mask)
            )

        return neg_loss_supcon
    

class Bin_SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf"""
    def __init__(self, temperature=0.07, base_temperature=0.07, adversarial=False,
                 n_views=2):
        super(Bin_SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.adversarial = adversarial
        self.n_views = n_views
        
    def forward(self, features, labels):
        '''
        features: a single Tensor.
        adversarial:
            * if False, this loss tries to make the features CORRELATED to labels.
            * if True, this loss tries to make the features INDEPENDENT of labels.
        '''
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze()
       
        pitch_label = labels[:, 0]
        pitch_label = pitch_label.unsqueeze(1)

        neg_loss_supcon_pitch = torch.mean(compute_supcon_loss(features, pitch_label, adversarial=self.adversarial,
            temperature=self.temperature, n_views=self.n_views, base_temperature=self.base_temperature))

        neg_loss_supcon = neg_loss_supcon_pitch

        return neg_loss_supcon
    

class Bin_SupConLoss_v2(nn.Module):
    '''
    by LZH. addresses pitch label discrepancies across datasets.
    Requires bin label + dataset ID in forward().
    '''
    def __init__(self, temperature=0.07, base_temperature=0.07, adversarial=False, nbins_pitch=9):
        super(Bin_SupConLoss_v2, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.adversarial = adversarial

        self.nbins_pitch = nbins_pitch

    def forward(self, features, labels):
        '''
        features: a single Tensor.
        adversarial:
            * if False, this loss tries to make the features CORRELATED to labels.
            * if True, this loss tries to make the features INDEPENDENT of labels.
        '''
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze()

        bin_labels, dataset_ids = labels
       
        pitch_label = bin_labels[:, 0]
        pitch_label = pitch_label.unsqueeze(1)

        neg_loss_supcon_pitch = torch.mean(compute_supcon_loss(features, pitch_label, adversarial=self.adversarial,
            temperature=self.temperature, base_temperature=self.base_temperature, dataset_ids=dataset_ids, nbins_pitch=self.nbins_pitch))

        neg_loss_supcon = neg_loss_supcon_pitch

        return neg_loss_supcon
    