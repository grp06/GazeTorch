'''
Edit Config_base based on input args. (for train mode)
'''
import argparse

def add_arguments(parser:argparse.ArgumentParser):
    # single_gpu: if True, GazeTorch will train on GPU #0.
    # use CUDA_VISIBLE_DEVICES="#" to specify which GPU to use.
    parser.add_argument("--single_gpu", action='store_true')
    parser.set_defaults(single_gpu=False)

    # cpu: if True, GazeTorch will train on CPU. Overrides `single_gpu`.
    parser.add_argument("--cpu", action='store_true')
    parser.set_defaults(cpu=False)

    # cv: cross-validation setting.
    parser.add_argument("--cv", action='store_true')
    parser.set_defaults(cv=False)

    # mpiiface-exclusive
    parser.add_argument('--mpii_fold_id', type=int, default=1)

    # training debug: if True, set debug=True for all datasets
    parser.add_argument("--debug", action='store_true')
    parser.set_defaults(debug=False)

    # parser.add_argument("-s", "--scl_factor", type=float)
    # Mehdi created this for testing different coefficients conveniently.
    # To do the same thing here, change the config file.

def modify_cfg(cfg, args):
    # single-gpu / cpu
    if args.single_gpu:
        cfg.train.gpus = [0]
    if args.cpu:
        cfg.train.gpus = 'cpu'

    # cross-validation
    cfg.train.put('cv', args.cv)

    # mpiiface-exclusive
    if hasattr(cfg.data.dataset_train.args, "val_subjects"):
        cfg.data.dataset_train.args.val_subjects = [args.mpii_fold_id]
        cfg.data.dataset_val.args.val_subjects = [args.mpii_fold_id]

    # debug
    if args.debug:
        cfg.data.dataset_train.args.debug = True
        cfg.data.dataset_val.args.debug = True

    return cfg