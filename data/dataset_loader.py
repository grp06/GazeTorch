from tools.printer import Color_string
import importlib

_private_imports = {
    "MPIIGAZE_eyes": "data.dataset.MPII_eyes.Dataset_MPIIGAZE_eyes",
    "MPIIFace": "data.dataset.mpiiface.Dataset_MPIIFace",
    # "XGaze": Dataset_XGaze, deprecated,
    "GC3d": "data.dataset.gc3d.Dataset_GC3d",
    "XGaze_v2": "data.dataset.xgaze.Dataset_XGaze",  # used Dataset_XGaze instead of Dataset_XGaze_v2
    "XGaze_v2_single": "data.dataset.xgaze.Dataset_XGaze", # used Dataset_XGaze instead of Dataset_XGaze_singleeye
    "Eyediap": "data.dataset.eyediap.Dataset_Eyediap",
    "Gaze360": "data.dataset.gaze360.Dataset_Gaze360",
    "GazeGene": "data.dataset.gazegene.Dataset_GazeGene",
    "RT_GENE": "data.dataset.rt_gene.Dataset_RT_GENE",
    "XGHybrid": "data.dataset.hybrid.Dataset_XplusG",
    "XG360Hybrid": "data.dataset.hybrid_triple.Dataset_XG360",
    "XGNHybrid": "data.dataset.hybrid_triple.Dataset_HybridXGN",
    "ZeroGaze_Vanilla": "data.dataset.zerogaze_vanilla.Dataset_ZeroGaze_Vanilla",
    "ZeroGaze_Glasses": "data.dataset.zerogaze_glasses.Dataset_ZeroGaze_Glasses",
    "ZeroGaze_Masks": "data.dataset.zerogaze_masks.Dataset_ZeroGaze_Masks",
    "Hybrid_v2": "data.dataset.hybrid_v2.Dataset_Hybrid_V2"
}


def get_dataset(cfg_dataset, dataset_base=None):
    '''
    dataset_base: if given, the new dataset instance will try to re-use some of the values within `dataset_base` instead of loading from storage again
    '''
    dataset_name = cfg_dataset.type
    if dataset_name in _private_imports:
        target_args = cfg_dataset.args if isinstance(cfg_dataset.args, dict) else cfg_dataset.args.__dict__
        target_args['dataset_base'] = dataset_base
        
        module_path, class_name = _private_imports[dataset_name].rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)(**target_args)
    else:
        raise Exception(Color_string.fail("[dataset_loader]Dataset not found: %s" % dataset_name))
