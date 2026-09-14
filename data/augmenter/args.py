from config.config_loader.config_loader import load_yaml
from data.consts import AUG_DEFAULT_ARGS_PATH_TEST

_NORMALIZE_KEYS = (
    'mean', 'std',
)

def arg_loader(default_args:dict, **kwargs):
    args = default_args.copy()
    for key, value in kwargs.items():
        args[key] = value
    return args

class AugmenterArgsHandler:
    def __init__(self, path_cfg=None) -> None:
        if path_cfg is None:
            path_cfg = AUG_DEFAULT_ARGS_PATH_TEST
        self.args = load_yaml(path_cfg) # A CfgNode instance
        
    def update_by_kwargs(self, in_place=True, **kwargs):
        if in_place:
            args = self.args
        else:
            args = self.args.copy()
        for key, value in kwargs.items():
            if key in _NORMALIZE_KEYS:
                args.normalize[key] = value
            else:
                args[key].switch = value
        return args