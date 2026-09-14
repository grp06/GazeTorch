'''
DataLoader_wrapper class
by Zhenhao. Initial version Sep 5 2022
Helps self-supervised setting when multiple views of a sample/minibatch is needed (is_training=True, num_views>1)
Otherwise, same as DataLoader.

For example, given a dataset returning (3,112,112) images and (2) labels, batch_size=8
When num_views == 1 (default):
    I return (8,3,112,112) and (8,2).
When num_views == 2,
    I return (16,3,112,112) and (16,2); where item #0 and #8 = item #0 in single-view case.

Feb '23: flip_every_other_view
If True, make view #(even) = not flipped and view #(odd) = flipped.
In this way each adjacent pair = flipped-and-non-flipped of a single sample, and we can apply symmetric loss therein.

To set num_views in a Config file, put "num_views" under data.dataset_train/val.args:
===Example===
dataset_train:
    type: XGaze_v2
    args:
      split: train
      num_views: 2
'''

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, DistributedSampler
from tools.distributed import Distributed_utils as du
import random
import gc
import copy


class FixedOrderSampler(Sampler): # works for 1-process setting
    def __init__(self, order_list):
        self._list = order_list
    
    def __len__(self):
        return len(self._list)
    
    def __iter__(self):
        return iter(range(len(self)))
    
    def __getitem__(self, index):
        return self._list[index]
    
class FixedOrderDistributedSampler(DistributedSampler):
    def __init__(self, dataset, order_list, num_replicas=None, rank=None, group_size=1):
        # Don't shuffle!
        super().__init__(dataset, num_replicas, rank, shuffle=False, drop_last=True)
        self._list = order_list
        self.group_size = group_size

    def __len__(self):
        # whole-group, whole-rank usable length (see __iter__)
        per_rank = (self.num_samples // self.group_size) * self.group_size
        return per_rank * self.num_replicas

    def __iter__(self):
        # Give this rank a CONTIGUOUS block instead of a stride, so each block of
        # `group_size` consecutive entries (a sample's n_views) stays on ONE rank.
        # A stride (indices[rank::num_replicas]) would split a sample's views across
        # different GPUs and destroy the contiguous-view layout the loss expects.
        total_size = self.num_samples * self.num_replicas
        indices = self._list[:total_size]

        # round each rank's share down to a whole number of groups
        per_rank = (self.num_samples // self.group_size) * self.group_size
        start = self.rank * per_rank
        indices = indices[start:start + per_rank]
        return iter(indices)

class DataLoader_wrapper:
    def __init__(self, dataset, batch_size, num_views=1,
                 split='train',
                 flip_every_other_view=False,
                 glasses_latter_half_views=False,
                 parallel=False,
                 share_t2d=False,
        ):
        '''
        n_view: # of views. (1 or 2)
        flip_every_other_view: if True, then view #(even)'s flip_mode = "off", view #(odd)'s flip_mode = "on"
        share_t2d: all views have the same Transform2D settings.
        '''
        self.n_views = num_views
        self.D = dataset
        self.D_len = len(dataset)
        self.batch_size = batch_size // num_views # the actual BS for each atomic DataLoader
        self.split = split

        self.hybrid = hasattr(self.D, 'hybrid')
        self.hybrid_hbn = hasattr(self.D, 'hybrid_hbn')

        if num_views>1 and flip_every_other_view:
            dataset_flipped = copy.deepcopy(dataset)
            if self.hybrid or self.hybrid_hbn:
                dataset.set_flip_mode('off')
                dataset_flipped.set_flip_mode('on')
            else:
                dataset.flip_mode = 'off'
                dataset_flipped.flip_mode = 'on'
            print("[DataLoader_wrapper] flip_every_other_view enabled.")

        # register datasets
        self.Ds = []
        for i in range(self.n_views):
            D = dataset
            if flip_every_other_view and i%num_views%2==1:
                D = dataset_flipped
            self.Ds.append(copy.copy(D))

        if glasses_latter_half_views:
            for i in range(self.n_views):
                self.Ds[i].aug = copy.deepcopy(self.Ds[i].aug)
                if i < self.n_views // 2:
                    self.Ds[i].aug.args.glasses.switch = False
                else:
                    # force glasses Aug
                    self.Ds[i].aug.args.glasses.switch = True
                    self.Ds[i].aug.args.glasses.p = 1.01
                self.Ds[i].update_aug_states()
            print("[DataLoader_wrapper] glasses_latter_half_views enabled.")

        self.share_t2d = share_t2d and num_views > 1
        if share_t2d:
            # need to manually refresh t2d at each __next__.
            # In this way, all samples in the MB have the same geometry Aug.
            self.D.aug.t2d_external = True
            for i in range(1, self.n_views):
                self.Ds[i].aug.t2d = self.D.aug.t2d
                self.Ds[i].aug.t2d_external = True

        self.parallel = parallel
        self.init_dataloader()

        self.fieldtype_X = [] # list(type of a field). can be "np", "tensor" or "etc".
        self.fieldtype_label = []

    def __iter__(self):
        self.iters = []
        for i in range(self.n_views):
            self.iters.append(iter(self.dls[i]))
        return self

    def __next__(self):
        if self.n_views==1:
            # same as a DataLoader
            return next(self.iters[0])
        else:
            if self.share_t2d:
                self.D.aug.t2d.generate_params()
            
            Xs_raw = []
            labels_raw = []
            for id_view in range(self.n_views):
                Xs, labels = next(self.iters[id_view])
                Xs_raw.append(Xs)
                labels_raw.append(labels)
            
            Xs_new = []
            labels_new = []
            for obj_raw, obj_new, ftype in zip(
                    [Xs_raw, labels_raw],
                    [Xs_new, labels_new], 
                    [self.fieldtype_X, self.fieldtype_label]
                ):
                if len(ftype)==0:
                    # count fields
                    for j, X in enumerate(obj_raw[0]):
                        if isinstance(X, np.ndarray):
                            ftype.append("np")
                        elif isinstance(X, torch.Tensor):
                            ftype.append("tensor")
                        elif isinstance(X, list):
                            ftype.append("list")
                        else:
                            ftype.append("etc")
                
                for id_f, f in enumerate(ftype):
                    if f=="np":
                        value = np.concatenate([obj_raw[id_view][id_f] for id_view in range(self.n_views)], axis=0)
                    elif f=="tensor":
                        value = torch.concat([obj_raw[id_view][id_f] for id_view in range(self.n_views)], axis=0)
                    elif f=="list":
                        value = [item for id_view in range(self.n_views) for item in obj_raw[id_view][id_f]]
                    else:
                        # LZH(25.12.24) NOTE: may not work for lists
                        value = []
                        map(value.extend, [obj_raw[id_view][id_f] for id_view in range(self.n_views)])
                    obj_new.append(value)
            return Xs_new, labels_new

    def __len__(self):
        return self.n_batches * self.n_views
    
    def synchronize(self):
        '''
        For multi-view - ensure that all views have the same ID sequence.
        Do this after running dataset.shuffle().
        '''
        if self.parallel and not du.is_master_proc():
            return
        for i in range(1, self.n_views):
            if self.hybrid:
                self.Ds[i].DX.id_list = self.Ds[0].DX.id_list.copy()
                self.Ds[i].DG.id_list = self.Ds[0].DG.id_list.copy()
            elif self.hybrid_hbn:
                self.Ds[i].DX.id_list = self.Ds[0].DX.id_list.copy()
                self.Ds[i].DG.id_list = self.Ds[0].DG.id_list.copy()
                self.Ds[i].DN.id_list = self.Ds[0].DN.id_list.copy()
            else:
                self.Ds[i].id_list = self.Ds[0].id_list.copy()
        print("[DataLoader_wrapper] All views' ID order synchronized.")
    
    def init_dataloader(self):
        '''
        Reset the DataLoaders.
        * in val mode, define a sampler as usual.

        Notes: we used to have this in trainer.py:
        if self.parallel:
            self.sampler_train.set_epoch(epoch)
        Seems this is no longer necessary as we have a new random `order_list` at each epoch.
        '''
        # keep a sample's n_views (laid out contiguously by Dataset_Hybrid_V2) together
        # through both the per-epoch shuffle AND the per-rank split.
        group_size = getattr(self.D, 'n_views', 1)
        if self.split == 'train':
            order_list = du.generate_order_list(self.D_len, parallel=self.parallel,
                                                group_size=group_size)
            if self.parallel:
                # In this case we have respective DLs for each process
                world_size = du.get_world_size()
                rank = du.get_rank()
            else:
                sampler = FixedOrderSampler(order_list)
                self.sampler = sampler
        else:
            if self.parallel:
                sampler = DistributedSampler(self.D, shuffle=False)
                self.sampler = sampler
            else:
                from torch.utils.data import SequentialSampler
                sampler = SequentialSampler(self.D)
                self.sampler = sampler

        self.dls = []
        self.iters = []
        for D in self.Ds:
            if self.split == 'train' and self.parallel:
                sampler = FixedOrderDistributedSampler(D, order_list, world_size, rank,
                                                       group_size=group_size)
            dl = DataLoader(D, self.batch_size, sampler=sampler)
            self.dls.append(dl)
            self.iters.append(iter(dl))
        self.n_batches = len(self.dls[0])

        gc.collect()
        print(f"[DataLoader_wrapper] DataLoader_{self.split} ready.")