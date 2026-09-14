'''
HybridGridDataset - shared base class for the V2 sub-datasets used by
Dataset_Hybrid_V2 (see dataset/hybrid_v2.py).

It pulls out the boilerplate that was duplicated across the three subs
(xgaze / gc3d / gazegene):
  - bin (pitch/yaw cid) computation on the hybrid grid -- NOT clipped, so that
    out-of-grid samples can be dropped by _filter_bin_in_range();
  - valid_ids assembly: dataset-specific filtering (subclass) followed by the
    in-grid filter;
  - the Dataset protocol (__len__, __getitem__) and shuffle().

Filtering stays in the sub; ALL sampling decisions live in Dataset_Hybrid_V2.

A subclass must implement two methods:
  _filter_dataset_specific(self) -> list[int]
      candidate gt indices that pass the dataset's own filters (split, cam,
      subject, image existence, per-subject balancing, ...). No sampler, no clip.
  _get(self, gt_idx, return_dict=False)
      the core of the original __getitem__ (augment / mixup / flip / eye loading
      / format conversion). Hybrid calls this directly with a gt index.
'''
import random
import numpy as np
from torch.utils.data import Dataset


# Which column of gt_data['headpose'] is yaw. Confirmed from the gt files: every
# dataset stores headpose as (pitch, yaw[, roll]) in radians, so yaw is column 1.
# (X/C are 2-col; N/G still carry a 3rd roll column -- indexing [1] works for both.)
HEADPOSE_YAW_COL = 1


class HybridGridDataset(Dataset):

    # Stamped by Dataset_Hybrid_V2 from its REGISTRY (sub.dataset_id = spec.dataset_id).
    # Stays None if a sub is used standalone without the hybrid setting it.
    dataset_id = None

    def _setup_bins_and_valid_ids(self, bin_args, gaze_array, headpose_array=None):
        '''Call once at the end of each sub's __init__.

        bin_args:      {'pitch_nbins', 'yaw_nbins', 'pitch_bias', ['bin_size'],
                        ['headpose_nbins', 'bin_size_hp']}
                       the hybrid grid definition (identical across all subs).
                       If 'headpose_nbins' is present, a 3rd binning axis on
                       headpose-yaw is added (centered at 0, own bin_size_hp).
        gaze_array:    full gt gaze array in radians, shape [N, 2] = (pitch, yaw),
                       indexed by gt index.
        headpose_array: full gt headpose array in radians, shape [N, 2 or 3]
                       (pitch, yaw[, roll]); only required when 'headpose_nbins'
                       is in bin_args. We bin on the yaw column only.
        '''
        self.nbins_pitch = bin_args['pitch_nbins']
        self.nbins_yaw   = bin_args['yaw_nbins']
        self.pitch_bias  = bin_args['pitch_bias']
        self.bin_size    = bin_args.get('bin_size', 4)
        self.bin_args    = bin_args
        self.bin         = True   # the hybrid grid always exists for V2 subs

        gaze = np.asarray(gaze_array)
        pitch_min = self.pitch_bias - self.nbins_pitch * self.bin_size / 2
        yaw_min   = 0               - self.nbins_yaw   * self.bin_size / 2
        # NOTE: do NOT clip -- out-of-grid cids must stay out of range so that
        # _filter_bin_in_range() can drop them (vs. the old datasets, which
        # np.clip()-ed and silently folded them onto the edge bins).
        self.pitch_cid = ((gaze[:, 0] * 180 / np.pi - pitch_min) // self.bin_size).astype(int)
        self.yaw_cid   = ((gaze[:, 1] * 180 / np.pi - yaw_min)   // self.bin_size).astype(int)

        # ---- optional 3rd axis: headpose yaw (centered at 0, own bin_size_hp) ----
        self.nbins_headpose = bin_args.get('headpose_nbins', None)   # None => 2D grid
        self.use_hp_bin     = self.nbins_headpose is not None
        if self.use_hp_bin:
            assert headpose_array is not None, \
                "bin_args has 'headpose_nbins' but no headpose array was passed"
            self.bin_size_hp = bin_args.get('bin_size_hp', self.bin_size)
            hp_yaw = np.asarray(headpose_array)[:, HEADPOSE_YAW_COL]
            hp_min = 0 - self.nbins_headpose * self.bin_size_hp / 2
            # NOT clipped, same as pitch/yaw -- out-of-grid samples get dropped.
            self.headpose_cid = ((hp_yaw * 180 / np.pi - hp_min) // self.bin_size_hp).astype(int)

        # gaze clip limits used by _get (same convention as the original datasets)
        self.limit_pitch_u = (self.pitch_bias + self.nbins_pitch / 2 * self.bin_size) * np.pi / 180
        self.limit_pitch_l = (self.pitch_bias - self.nbins_pitch / 2 * self.bin_size) * np.pi / 180
        self.limit_yaw_u   = ( self.nbins_yaw / 2) * self.bin_size * np.pi / 180
        self.limit_yaw_l   = (-self.nbins_yaw / 2) * self.bin_size * np.pi / 180

        candidates = self._filter_dataset_specific()
        self.valid_ids = self._filter_bin_in_range(candidates)
        self.num_data = len(self.valid_ids)
        print(f"[{type(self).__name__}] split={getattr(self, 'split', '?')}  "
              f"candidates={len(candidates)}  valid_ids={len(self.valid_ids)}")

    def _filter_bin_in_range(self, candidates):
        '''Keep only candidates whose bin cids fall inside the grid (pitch, yaw,
        and -- when enabled -- headpose). Duplicates in `candidates` (e.g. GC3D
        per-subject balancing) are preserved.'''
        out = []
        for i in candidates:
            if not (0 <= self.pitch_cid[i] < self.nbins_pitch): continue
            if not (0 <= self.yaw_cid[i]   < self.nbins_yaw):   continue
            if self.use_hp_bin and not (0 <= self.headpose_cid[i] < self.nbins_headpose):
                continue
            out.append(i)
        return out

    # ===================================================================
    # Dataset protocol
    # ===================================================================
    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, i, return_dict=False):
        '''Standalone entry point; `i` indexes into valid_ids.
        (Hybrid bypasses this and calls _get(gt_idx) directly.)'''
        return self._get(self.valid_ids[i], return_dict=return_dict)

    def shuffle(self):
        '''Only re-orders valid_ids; no resampling (sampling lives in the hybrid).'''
        random.shuffle(self.valid_ids)

    # ===================================================================
    # Abstract -- implemented by each sub
    # ===================================================================
    def _filter_dataset_specific(self):
        raise NotImplementedError

    def _get(self, gt_idx, return_dict=False):
        raise NotImplementedError
