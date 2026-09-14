'''
Dataset_Hybrid_V2 - hybrid dataset for XGaze + GC3D + GazeGene.

Key differences vs Dataset_HybridXGN:
  - bin uniform on the hybrid grid (not dataset uniform)
  - sparse bins are dropped entirely (no quota-filling by repetition)
  - does not hold "sub datasets that own a sampler"; after refactor the subs
    only do loading + filtering, and ALL sampling decisions live in this
    class's _sample_binuniform method.

The sub datasets are constructed INSIDE this class's __init__ from a single shared
config (so training code can do Dataset_Hybrid_V2(**config_args)). Each sub must
satisfy this contract:
  attributes:
    sub.valid_ids:   list[int]        gt indices that pass all filters AND
                                      fall inside the hybrid grid
    sub.pitch_cid:   np.ndarray[int]  pitch bin id on the hybrid grid, per gt index
    sub.yaw_cid:     np.ndarray[int]  yaw bin id on the hybrid grid, per gt index
    sub.nbins_pitch: int              for grid-consistency assertion
    sub.nbins_yaw:   int              same
    sub.use_hp_bin:  bool             whether the optional 3rd (headpose-yaw) axis
                                      is active (set from bin_args['headpose_nbins'])
    sub.headpose_cid:np.ndarray[int]  headpose-yaw bin id, per gt index (only when
                                      use_hp_bin)
    sub.nbins_headpose: int           headpose-axis bin count (only when use_hp_bin)
  methods:
    sub._get(gt_idx, return_dict=True) -> dict
        contains all augment/flip/mixup/eye loading logic
        (the core of the original sub.__getitem__).
        When return_dict=True, returns a {key: value} dict.
'''
from typing import Optional
from dataclasses import dataclass, field
import itertools
import random
import numpy as np
from torch.utils.data import Dataset

from data.hybrid_v2.base import HEADPOSE_YAW_COL


@dataclass(frozen=True)
class SubSpec:
    '''One hybrid sub-dataset, keyed by its single-letter tag:

        tag -> (dataset_id, class, default knobs)

    dataset_id: integer id (used when 'dataset_id' is in inputs/labels). This is
                the single source of truth: the hybrid stamps it onto each built
                sub (sub.dataset_id) and derives tag_to_id from it.
    cls:        the sub Dataset class; None = recognized tag but not wired yet
                (e.g. 'G' Gaze360).
    knobs:      per-dataset default constructor kwargs, overridable via sub_knobs.
    '''
    dataset_id: int
    cls: type
    knobs: dict = field(default_factory=dict)


class Dataset_Hybrid_V2(Dataset):

    def __init__(self,
                 bin_args: dict,
                 target_N: int,                    # per_bin_N is DERIVED from this + bin_args
                 split: str = 'train',
                 inputs: list = ('face',),
                 labels: list = ('gaze',),
                 tags: tuple = ('X', 'C', 'N'),    # which subs to build
                 # ---- shared config forwarded to every sub ----
                 augment: bool = True,
                 mixup: bool = False, mixup_halfandhalf: bool = False, mixup_thresh: float = 0.0,
                 flip_mode: str = 'off',
                 augmenter_cfg_path: Optional[str] = None,   # None -> each sub uses its own yaml
                 augmenter_options: Optional[dict] = None,
                 skip_val: bool = False,
                 debug: bool = False,
                 rgb_no_normalization: bool = False, return_tensor: bool = False,
                 img_channel_chw: bool = True,
                 # ---- per-dataset knobs: {tag: {kwarg: value}}, overrides REGISTRY ----
                 sub_knobs: Optional[dict] = None,
                 # ---- hybrid-only ----
                 tag_to_id: Optional[dict] = None,
                 ssl_clique_size: int = 1,
                 n_views = 1,
                 **legacy):                        # swallow + log legacy config keys
        '''
        bin_args:    {'pitch_nbins', 'yaw_nbins', 'pitch_bias', 'bin_size'}
                     definition of the hybrid grid. Forwarded to every sub.
                     OPTIONAL 3rd axis: add 'headpose_nbins' (and optionally
                     'bin_size_hp', default = bin_size) to bin on headpose-yaw too,
                     making the grid (pitch, yaw, headpose_yaw) and the 'bin' field
                     length-3. Absent -> plain 2D grid (unchanged behaviour).
        target_N:    nominal total sample count. per_bin_N is derived from it:
                       train     -> target_N // (pitch_nbins * yaw_nbins * H)
                       non-train -> target_N // ((pitch_nbins-2) * (yaw_nbins-2) * H)
                     where H = headpose_nbins if the 3rd axis is on, else 1.
                     Bins with fewer candidates than per_bin_N are dropped entirely,
                     so the realized len() is kept_bins * per_bin_N <= target_N.
        split:       'train' / 'val' / 'test'; non-train drops the outer ring of bins.
        inputs/labels: same IO field lists as the old datasets; forwarded to subs.
        tags:        which sub datasets to build, subset of REGISTRY keys
                     (e.g. ('X','C','N')). Selects set membership only.
        augment/mixup*/flip_mode/augmenter_*/skip_val/debug/rgb_no_normalization/
        return_tensor/img_channel_chw: shared config forwarded to every sub.
        sub_knobs:   per-dataset constructor-kwarg overrides, {tag: {kwarg: value}}.
                     Merged over each sub's REGISTRY default knobs (e.g.
                     {'X': {'cam_id': [0, 1]}, 'C': {'n_per_person': 300}}).
                     Keys for tags not in `tags` are ignored. None -> registry
                     defaults for every selected sub.
        tag_to_id:   tag string -> dataset_id integer mapping
                     (only used when 'dataset_id' is in inputs/labels).
        ssl_clique_size: if > 1, post-process each epoch's sample list so that
                     same-subject GC3D ('G') samples are grouped into cliques of
                     this size (method A: bin-uniform first, then cluster). 1 = off.
                     NOTE: to make cliques survive into mini-batches, the DataLoader
                     must use shuffle=False and a batch_size that is a multiple of
                     ssl_clique_size.
        **legacy:    leftover config keys (bin_sampler, facemap, filter, ...) that no
                     longer map to params; swallowed and logged, not used.
        '''
        super().__init__()
        if legacy:
            print(f"[Dataset_Hybrid_V2] ignoring legacy config keys: {sorted(legacy)}")

        self.ssl_clique_size = ssl_clique_size
        self.inputs = list(inputs)
        self.labels = list(labels)
        self.all    = self.inputs + self.labels
        self.split  = split
        self.flip_mode = flip_mode

        # ---- bin grid params ----
        self.nbins_pitch = bin_args['pitch_nbins']
        self.nbins_yaw   = bin_args['yaw_nbins']
        # optional 3rd axis on headpose-yaw; None => plain 2D grid
        self.nbins_hpyaw = bin_args.get('headpose_nbins', None)
        self.use_hp_bin  = self.nbins_hpyaw is not None

        self.pitch_bias  = bin_args['pitch_bias']
        self.bin_size    = bin_args.get('bin_size', 4)
        self.bin_size_hp = bin_args.get('bin_size_hp', self.bin_size)
        self.bin_args    = bin_args

        # the recompute in __getitem__ needs the post-_get headpose value
        if self.use_hp_bin and 'bin' in self.all and 'headpose' not in self.all:
            raise ValueError("headpose binning with 'bin' requested also needs "
                             "'headpose' in inputs/labels (to recompute bin[2]).")

        # ---- derive per_bin_N from target_N + grid (old sampler bin_quote rule) ----
        hp = self.nbins_hpyaw if self.use_hp_bin else 1
        usable = (self.nbins_pitch * self.nbins_yaw * hp) if split == 'train' \
            else ((self.nbins_pitch - 2) * (self.nbins_yaw - 2) * hp)
        self.per_bin_N = target_N // usable

        # ---- build the shared kwargs forwarded to every sub ----
        common = dict(split=split, augment=augment, debug=debug,
                      inputs=list(inputs), labels=list(labels),
                      mixup=mixup, mixup_halfandhalf=mixup_halfandhalf, mixup_thresh=mixup_thresh,
                      flip_mode=flip_mode, augmenter_options=(augmenter_options or {}),
                      skip_val=skip_val, bin_args=bin_args,
                      rgb_no_normalization=rgb_no_normalization, return_tensor=return_tensor,
                      img_channel_chw=img_channel_chw,
                      child_of_hybrid=True)   # so subject_cid offsets apply across datasets
        from data.augmenter.augmenter import initialize_augmenter
        augment_split = 'train'
        import data.consts as consts
        augmenter_dir=consts.AUG_DIR_MATERIALS
        self.aug = initialize_augmenter(augment_split, augmenter_dir, augmenter_cfg_path)

        if augmenter_cfg_path is not None:
            common['augmenter_cfg_path'] = augmenter_cfg_path

        # ---- REGISTRY: single source of truth for tag -> (dataset_id, class, knobs) ----
        from data.hybrid_v2.xgaze    import Dataset_XGaze_V2
        from data.hybrid_v2.gc3d     import Dataset_GC3d_V2
        from data.hybrid_v2.gazegene import Dataset_GazeGene_V2
        from data.hybrid_v2.gaze360  import Dataset_Gaze360_V2
        REGISTRY = {
            'X': SubSpec(0, Dataset_XGaze_V2,    {}), # should use all cam_id
            'C': SubSpec(1, Dataset_GC3d_V2,     {'n_per_person': 500}),
            'N': SubSpec(2, Dataset_GazeGene_V2, {}),
            'G': SubSpec(3, Dataset_Gaze360_V2,  {}),   # Gaze360
        }

        bad = [t for t in tags if t not in REGISTRY]
        assert not bad, f"unknown tags {bad}; known: {sorted(REGISTRY)}"

        # tag -> integer dataset_id, derived from REGISTRY (overridable via tag_to_id arg)
        self.tag_to_id = tag_to_id or {t: s.dataset_id for t, s in REGISTRY.items()}

        # ---- construct subs from the registry (per-dataset knobs overridable) ----
        self.subs = {}
        for t in tags:
            spec = REGISTRY[t]
            assert spec.cls is not None, f"sub '{t}' is in REGISTRY but not wired yet"
            knobs = {**spec.knobs, **(sub_knobs.get(t, {}) if sub_knobs else {})}
            sub = spec.cls(**knobs, **common)
            sub.dataset_id = spec.dataset_id   # single source of truth for the id
            self.subs[t] = sub

        # ---- assert: every sub uses the same hybrid grid ----
        for tag, sub in self.subs.items():
            assert sub.nbins_pitch == self.nbins_pitch, \
                f"sub '{tag}' nbins_pitch={sub.nbins_pitch} != hybrid {self.nbins_pitch}"
            assert sub.nbins_yaw == self.nbins_yaw, \
                f"sub '{tag}' nbins_yaw={sub.nbins_yaw} != hybrid {self.nbins_yaw}"
            assert sub.use_hp_bin == self.use_hp_bin, \
                f"sub '{tag}' use_hp_bin={sub.use_hp_bin} != hybrid {self.use_hp_bin}"
            if self.use_hp_bin:
                assert sub.nbins_headpose == self.nbins_hpyaw, \
                    f"sub '{tag}' nbins_headpose={sub.nbins_headpose} != hybrid {self.nbins_hpyaw}"

        self.set_flip_mode(flip_mode)

        # ---- build the pool ----
        # dict keyed by the bin tuple (p, y) or (p, y, h); unifies 2D / 3D grids.
        self.pool = {}
        for tag, sub in self.subs.items():
            for gt_idx in sub.valid_ids:
                p = int(sub.pitch_cid[gt_idx])
                y = int(sub.yaw_cid[gt_idx])
                # sub guarantees the cids are inside [0, nbins) on every axis
                if self.use_hp_bin:
                    key = (p, y, int(sub.headpose_cid[gt_idx]))
                else:
                    key = (p, y)
                self.pool.setdefault(key, []).append((tag, int(gt_idx)))

        # ---- histogram (for debugging: how many candidates each bin has) ----
        hist_shape = (self.nbins_pitch, self.nbins_yaw)
        if self.use_hp_bin:
            hist_shape = hist_shape + (self.nbins_hpyaw,)
        self.actual_bin_pop = np.zeros(hist_shape, dtype=int)
        for key, bucket in self.pool.items():
            self.actual_bin_pop[key] = len(bucket)

        # n_views must be set before the first sampling so __init__'s id_list
        # already carries the per-view flip flags (see _apply_views).
        self.n_views = n_views

        # ---- first sampling ----
        base = self._sample_binuniform()

        # gaze range limits used to clip gaze in __getitem__ (follows old datasets' convention)
        self.limit_pitch_u = (self.pitch_bias + self.nbins_pitch / 2 * self.bin_size) * np.pi / 180
        self.limit_pitch_l = (self.pitch_bias - self.nbins_pitch / 2 * self.bin_size) * np.pi / 180
        self.limit_yaw_u   = ( self.nbins_yaw / 2 * self.bin_size) * np.pi / 180
        self.limit_yaw_l   = (-self.nbins_yaw / 2 * self.bin_size) * np.pi / 180

        # kept_bins computed from the base (pre-view) sample list.
        print(f"[Dataset_Hybrid_V2] split={split} len={len(base)}  "
              f"target_N={target_N} per_bin_N={self.per_bin_N}  "
              f"kept_bins={len(base)//self.per_bin_N if self.per_bin_N else 0}")

        self.id_list = self._apply_views(base)

    # ===================================================================
    # Dataset protocol
    # ===================================================================
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, i, return_dict=False):
        tag, gt_idx, force_flip = self.id_list[i]

        # Key step: call the sub's _get(gt_idx) so all augment/mixup/eye loading
        # logic from the sub flows through unchanged.
       
        values = self.subs[tag]._get(gt_idx, return_dict=True, force_flip=force_flip)

        # Datasets store headpose as (pitch, yaw) [X, C] or (pitch, yaw, roll) [G, N].
        # The ragged length breaks the default collate on a mixed batch. roll is unused
        # (binning + viz only read HEADPOSE_YAW_COL), so collapse to (pitch, yaw).
        # Slice (not np.asarray) to preserve tensor/list type under return_tensor.
        if 'headpose' in values and values['headpose'] is not None:
            values['headpose'] = values['headpose'][:2]

        # hybrid-level extra fields
        if 'dataset_id' in self.all:
            values['dataset_id'] = self.tag_to_id[tag]

        if 'headpose' in self.all and len(values['headpose'])==2:
            values['headpose'] = np.append(values['headpose'], 0)

        # Recompute bin on the hybrid grid (do NOT trust the sub's bin --
        # the sub uses its own grid, plus mixup/flip in _get may have changed gaze).
        if 'bin' in self.all:
            g = values['gaze']
            pitch_min = self.pitch_bias - self.nbins_pitch * self.bin_size / 2
            yaw_min   = 0               - self.nbins_yaw   * self.bin_size / 2
            p = int(np.clip(
                (g[0] * 180 / np.pi - pitch_min) // self.bin_size,
                0, self.nbins_pitch - 1))
            y = int(np.clip(
                (g[1] * 180 / np.pi - yaw_min)   // self.bin_size,
                0, self.nbins_yaw - 1))
            if self.use_hp_bin:
                # 3rd axis: headpose yaw (recomputed from post-_get headpose, which
                # flip/mixup may have changed). Centered at 0, own bin_size_hp.
                hp = values['headpose']
                hp_min = 0 - self.nbins_hpyaw * self.bin_size_hp / 2
                h = int(np.clip(
                    (hp[HEADPOSE_YAW_COL] * 180 / np.pi - hp_min) // self.bin_size_hp,
                    0, self.nbins_hpyaw - 1))
                values['bin'] = np.array([p, y, h])
            else:
                values['bin'] = np.array([p, y])

        # clip gaze to bin range (follows old datasets' convention)
        if 'gaze' in values:
            g = values['gaze']
            g[0] = np.clip(g[0], self.limit_pitch_l, self.limit_pitch_u)
            g[1] = np.clip(g[1], self.limit_yaw_l,   self.limit_yaw_u)
            values['gaze'] = g

        if return_dict:
            return values

        Xs     = [values[k] for k in self.inputs]
        labels = [values[k] for k in self.labels]
        return Xs, labels

    def shuffle(self):
        '''Resample once per epoch: bin-uniform property preserved, but the
        specific samples picked per bin are re-drawn.'''
        self.id_list = self._apply_views(self._sample_binuniform())

    # ===================================================================
    # n_views expansion (incl. flip-views pairing)
    # ===================================================================
    def _apply_views(self, base_list):
        '''Expand the base (tag, gt_idx) sample list into per-view entries
        (tag, gt_idx, force_flip). force_flip:
            None       -> the sub's _get decides flip itself (its flip_mode logic)
            True/False -> flip is pinned for this view.

        With flip_mode=='on' and n_views>1, views alternate original/flipped
        contiguously per sample (e.g. n_views=4 -> orig, flip, orig, flip), so a
        sample's views stay adjacent in id_list. Requires even n_views.
        Other flip_modes ('off'/'random') keep the plain duplicate behaviour
        (force_flip=None per view).'''
        if self.n_views <= 1:
            return [(tag, gt_idx, None) for (tag, gt_idx) in base_list]

        if self.flip_mode == 'on':
            assert self.n_views % 2 == 0, \
                f"flip_mode='on' with n_views>1 requires even n_views, got {self.n_views}"
            pattern = [False, True] * (self.n_views // 2)
        else:
            pattern = [None] * self.n_views

        out = []
        for (tag, gt_idx) in base_list:
            for f in pattern:                 # contiguous per sample
                out.append((tag, gt_idx, f))
        print(f"[Dataset_Hybrid_V2] n_views={self.n_views} flip_mode={self.flip_mode} "
              f"-> id_list expanded {len(base_list)} -> {len(out)}")
        return out

    # ===================================================================
    # Core: bin-uniform sampling with drop-sparse rule
    # ===================================================================
    def _sample_binuniform(self):
        out = []
        # iterate every grid bin: (pitch, yaw) or (pitch, yaw, headpose) when active
        dims = [range(self.nbins_pitch), range(self.nbins_yaw)]
        if self.use_hp_bin:
            dims.append(range(self.nbins_hpyaw))
        for key in itertools.product(*dims):
            p, y = key[0], key[1]
            # Non-train: drop the outer ring of bins (gaze pitch/yaw only; follows
            # old sampler convention). The headpose axis is not ring-dropped.
            if self.split != 'train' and (
                p in (0, self.nbins_pitch - 1) or
                y in (0, self.nbins_yaw   - 1)):
                continue
            bucket = self.pool.get(key, [])
            if len(bucket) < self.per_bin_N:
                continue                              # <-- drop sparse bin
            out.extend(random.sample(bucket, self.per_bin_N))

        # SSL post-process (method A): group same-subject GC3D samples into cliques.
        if self.ssl_clique_size > 1:
            return self._cluster_by_subject(out, self.ssl_clique_size)

        random.shuffle(out)
        return out

    def _cluster_by_subject(self, id_list, clique_size):
        '''Best-effort: group same-(tag='G', subject) samples into cliques so that
        nearby mini-batches contain multiple images of the same subject. Other
        datasets' samples are treated as singleton cliques. Bin-uniformity is
        preserved (this only reorders the already-sampled list).'''
        by_subject = {}
        others = []
        for tag, gt_idx in id_list:
            if tag == 'G' and 'G' in self.subs:
                subj = self.subs['G'].gt_data['fn'][gt_idx][:5]
                by_subject.setdefault(subj, []).append((tag, gt_idx))
            else:
                others.append((tag, gt_idx))

        cliques = []
        for subj, samples in by_subject.items():
            random.shuffle(samples)
            for i in range(0, len(samples), clique_size):
                cliques.append(samples[i:i + clique_size])
        for s in others:
            cliques.append([s])

        random.shuffle(cliques)
        out = []
        for c in cliques:
            out.extend(c)
        return out

    # ===================================================================
    # Hybrid-level convenience: forward calls to all subs
    # ===================================================================
    def set_flip_mode(self, flip_mode: str):
        self.flip_mode = flip_mode
        for sub in self.subs.values():
            sub.flip_mode = flip_mode

    def update_aug_states(self):
        for sub in self.subs.values():
            if hasattr(sub, 'update_aug_states'):
                sub.update_aug_states()


# =======================================================================
# Self-test: optional headpose (3rd) bin dimension
# =======================================================================
def _selftest_headpose_bin(target_N=40000, tags=('X', 'C', 'N', 'G'),
                           augmenter_cfg_path=None):
    '''Exercise the optional headpose-yaw 3rd bin axis end to end. Run on the box:
        SERVER=<n> python -m data.dataset.hybrid_v2
    Checks: bin shape, in-range cids, gaze/headpose->bin recompute cross-check,
    per-bin uniformity, and the 2D back-compat path. Prints a PASS summary.'''
    from collections import Counter

    BA3 = {'pitch_nbins': 15, 'yaw_nbins': 17, 'pitch_bias': -12, 'bin_size': 4,
           'headpose_nbins': 9, 'bin_size_hp': 10}   # hp axis: +/-45 deg, 9 bins
    common = dict(tags=tags, split='train', augment=False, n_views=1,
                  flip_mode='off', ssl_clique_size=1,
                  inputs=['gaze'], labels=['headpose', 'bin'])
    if augmenter_cfg_path is not None:
        common['augmenter_cfg_path'] = augmenter_cfg_path

    print("\n===== self-test: headpose 3rd bin dimension =====")
    ds = Dataset_Hybrid_V2(bin_args=BA3, target_N=target_N, **common)
    assert ds.use_hp_bin and ds.nbins_hpyaw == 9, "3D grid not active"

    # --- per-bin uniformity over the 3D keys, rebuilt from the sub cids ---
    keycount = Counter()
    for (tag, gt_idx, _f) in ds.id_list:
        sub = ds.subs[tag]
        keycount[(int(sub.pitch_cid[gt_idx]), int(sub.yaw_cid[gt_idx]),
                  int(sub.headpose_cid[gt_idx]))] += 1
    assert len(keycount) > 0, "no bins kept -- raise target_N"
    bad = {k: c for k, c in keycount.items() if c != ds.per_bin_N}
    assert not bad, f"non-uniform bins (want {ds.per_bin_N}): {list(bad.items())[:5]}"
    print(f"  uniformity OK: {len(keycount)} bins x per_bin_N={ds.per_bin_N}")

    # --- shape + independent gaze/headpose -> bin recompute on a sample ---
    pmin = BA3['pitch_bias'] - BA3['pitch_nbins'] * BA3['bin_size'] / 2
    ymin = 0 - BA3['yaw_nbins'] * BA3['bin_size'] / 2
    hmin = 0 - BA3['headpose_nbins'] * BA3['bin_size_hp'] / 2
    n = min(300, len(ds))
    for i in range(n):
        v = ds.__getitem__(i, return_dict=True)
        b = v['bin']
        assert tuple(b.shape) == (3,), f"bin shape {b.shape} != (3,)"
        assert 0 <= b[0] < 15 and 0 <= b[1] < 17 and 0 <= b[2] < 9, f"bin out of range: {b}"
        # (a) independent recompute from the returned gaze/headpose
        g, hp = v['gaze'], v['headpose']
        p = int(np.clip((g[0] * 180 / np.pi - pmin) // BA3['bin_size'], 0, 14))
        y = int(np.clip((g[1] * 180 / np.pi - ymin) // BA3['bin_size'], 0, 16))
        h = int(np.clip((hp[HEADPOSE_YAW_COL] * 180 / np.pi - hmin) // BA3['bin_size_hp'], 0, 8))
        assert (p, y, h) == tuple(int(x) for x in b), f"recompute {(p, y, h)} != bin {tuple(b)}"
        # (b) flip_mode='off' + no mixup => __getitem__ bin must equal the sub's
        #     precomputed cids (the pool key from base.py): cross-checks the two paths.
        tag, gt_idx, _f = ds.id_list[i]
        sub = ds.subs[tag]
        expect = (int(sub.pitch_cid[gt_idx]), int(sub.yaw_cid[gt_idx]),
                  int(sub.headpose_cid[gt_idx]))
        assert expect == tuple(int(x) for x in b), f"sub cid {expect} != bin {tuple(b)}"
    print(f"  shape + recompute + base/hybrid cross-check OK on {n} items")

    # --- back-compat: no headpose_nbins -> plain 2D grid ---
    BA2 = {'pitch_nbins': 15, 'yaw_nbins': 17, 'pitch_bias': -12, 'bin_size': 4}
    ds2 = Dataset_Hybrid_V2(bin_args=BA2, target_N=target_N, **common)
    assert ds2.use_hp_bin is False, "2D run unexpectedly enabled headpose bin"
    assert tuple(ds2.__getitem__(0, return_dict=True)['bin'].shape) == (2,), "2D bin not length-2"
    print(f"  back-compat OK: 2D bin length-2, len={len(ds2)}")

    print("===== self-test PASS =====\n")
    return ds


# =======================================================================
# Visual self-test: does the headpose (3rd) bin axis really group by head-yaw?
# =======================================================================
def _selftest_headpose_visual(
        out_path="/store/samuel/debug/headpose_montage.png",
        per_bin=8, img_px=128, target_N=200000,
        tags=('X', 'C', 'N', 'G'),
        augmenter_cfg_path="/store/samuel/GazeTorch/config/temp_noaug.yaml"):
    '''
    Eyeball check for the headpose 3rd bin axis. Writes ONE montage image:
        each ROW    = one headpose-yaw bin h (top = most negative head-yaw,
                      bottom = most positive)
        each COLUMN = a random sample from that bin, gaze (pitch,yaw) left free
    If the axis is correct, heads rotate horizontally as you scan top->bottom,
    and within a row they all face the same way. Each face is labelled with its
    head-yaw in degrees; per-row mean head-yaw is printed and should increase
    monotonically with h.

    REQUIRES flip_mode='off' + n_views=1: a horizontal flip mirrors head-yaw,
    which would drop opposite-facing heads into the same row and ruin the check.
    Sampling is drawn straight from ds.pool (all candidates, pre drop-sparse), so
    bin-uniformity / per_bin_N do not matter here.
    '''
    import cv2, os
    import numpy as np
    from einops import rearrange

    BA3 = {'pitch_nbins': 16, 'yaw_nbins': 20, 'pitch_bias': -8, 'bin_size': 6.0,
           'headpose_nbins': 9, 'bin_size_hp': 10}     # hp axis: +/-45 deg, 9 bins
    ds = Dataset_Hybrid_V2(
        bin_args=BA3, target_N=target_N, tags=tags, split='train',
        inputs=['face'], labels=['headpose', 'gaze'],
        augment=False, rgb_no_normalization=True,
        flip_mode='off', n_views=1, ssl_clique_size=1,
        augmenter_cfg_path=augmenter_cfg_path,
    )
    assert ds.use_hp_bin, "headpose axis not active -- add 'headpose_nbins' to bin_args"

    # group ALL candidate samples by their headpose bin h (3rd field of the pool key)
    by_h = {}
    for key, bucket in ds.pool.items():
        by_h.setdefault(key[2], []).extend(bucket)

    def render(tag, gt_idx):
        v = ds.subs[tag]._get(gt_idx, return_dict=True, force_flip=False)
        img = np.asarray(v['face']).astype('float32')          # CHW, RGB
        C = img.shape[0]
        mn = img.reshape(C, -1).min(1)[:, None, None]
        mx = img.reshape(C, -1).max(1)[:, None, None]
        img = (img - mn) / (mx - mn + 1e-8)
        img = rearrange(img, 'c h w -> h w c')
        img = (img * 255).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)             # RGB -> BGR for imwrite
        img = cv2.resize(img, (img_px, img_px))
        hp_deg = float(v['headpose'][HEADPOSE_YAW_COL]) * 180 / np.pi
        cv2.putText(img, f"{hp_deg:+.0f}", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img, hp_deg

    blank = np.zeros((img_px, img_px, 3), dtype='uint8')
    rows, row_means = [], []
    for h in range(ds.nbins_hpyaw):
        bucket = list(by_h.get(h, []))
        random.shuffle(bucket)
        cells, degs = [], []
        for (tag, gt_idx) in bucket[:per_bin]:
            img, deg = render(tag, gt_idx)
            cells.append(img); degs.append(deg)
        row_means.append(float(np.mean(degs)) if degs else None)
        while len(cells) < per_bin:        # pad short rows so hstack lines up
            cells.append(blank)
        rows.append(np.hstack(cells))
    montage = np.vstack(rows)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = cv2.imwrite(out_path, montage)
    print(f"\n[headpose_visual] wrote {out_path}  ok={ok}  "
          f"rows(h)={ds.nbins_hpyaw} x cols={per_bin}")
    print("  per-row mean head-yaw (should increase monotonically with h):")
    for h, m in enumerate(row_means):
        print(f"    h={h}: {'(empty)' if m is None else f'{m:+.1f} deg'}")
    return ds


# =======================================================================
# Usage example (caller side)
# =======================================================================
if __name__ == '__main__':
    random.seed(42)

    # Visual check: montage with rows = head-yaw bins. Heads should rotate
    # horizontally top->bottom; per-row mean head-yaw should be monotonic.
    _selftest_headpose_visual()

    exit(1)

    

    # Verify the optional headpose 3rd bin axis (3D + 2D back-compat) first.
    _selftest_headpose_bin(
        augmenter_cfg_path="/store/samuel/GazeTorch/config/temp_noaug.yaml")

    BA = {
        'pitch_nbins': 15,
        'yaw_nbins':   17,
        'pitch_bias':  -12,
        'bin_size':    4,

    }

    # The hybrid builds the three subs itself and forwards this config to them.
    # For this debug dump we turn OFF augmentation + normalization so the saved
    # images show true colours (rgb_no_normalization=True -> raw 0-255 RGB).
    ds = Dataset_Hybrid_V2(
        bin_args             = BA,
        target_N             = 384000,   # per_bin_N is derived from this + bin_args
        inputs               = ['face'],
        labels               = ['subject_cid', 'gaze', 'bin', 'dataset_id', 'facemap_v3', 'wE_L', 'wI_L', 'wE_R', 'wI_R', 'id', 'aug_mask', 'aug_glasses'],
        split                = 'train',
        ssl_clique_size      = 1,        # optional; 1 = off
        augment              = False,        # no geometric/colour augmentation
        rgb_no_normalization = False,         # raw 0-255 RGB pixels (bypasses normalization)
        n_views              = 4,
        flip_mode            = "on",
        augmenter_cfg_path   = "/store/samuel/GazeTorch/config/temp_noaug.yaml"
    )

    import cv2
    import numpy as np
    from einops import rearrange
    import os

    output_dir = "/store/samuel/debug"
    os.makedirs(output_dir, exist_ok=True)   # cv2.imwrite silently no-ops if dir is missing


    ds.shuffle()

    '''
    rgb_no_normalization = True
    '''
    # for i in range(32):
    #     v = ds.__getitem__(i, return_dict=True)
    #     img = np.asarray(v['face'])                  # CHW, RGB, already 0-255

    #     img = rearrange(img, 'd h w -> h w d').astype('uint8')
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # RGB -> BGR for imwrite
    #     ok = cv2.imwrite(os.path.join(output_dir, f"{i}.png"), img)
    #     if not ok:
    #         print(f"[debug] cv2.imwrite FAILED for {i}.png")


    '''
    rgb_no_normalization = False
    '''
    for i in range(32):
        v = ds.__getitem__(i, return_dict=True)
        img = np.asarray(v['face']).astype('float32')
        
        C = img.shape[0]
        mn = img.reshape(C, -1).min(1)[:, None, None]
        mx = img.reshape(C, -1).max(1)[:, None, None]
        img = (img - mn) / (mx - mn + 1e-8)

        # img = np.clip(img * 0.25 + 0.5, 0, 1)

        img = rearrange(img, 'c h w -> h w c')
        img = (img * 255).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # RGB -> BGR for imwrite
        ok = cv2.imwrite(os.path.join(output_dir, f"{i}.png"), img)
        if not ok:
            print(f"[debug] cv2.imwrite FAILED for {i}.png")
