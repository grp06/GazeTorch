from model.module_base import Module_base
from tools.printer import Color_string
import importlib

_private_imports = {
    # backbone
    "mobilenet_alpha": "model.backbone.alpha_mobilenet_v2.Mobilenet_V2", 
    "mobilenet_v2": "model.backbone.mobilenetv2.mobilenet_v2",
    "mobilenet_v2_ca": "model.backbone.mobilenetv2.MobileNetV2_CA",
    "vgg11": "model.backbone.vgg.vgg11",
    "vgg16": "model.backbone.vgg.vgg16",
    "vgg16_bn": "model.backbone.vgg.vgg16_bn",
    "vgg19": "model.backbone.vgg.vgg19",
    "vgg19_bn": "model.backbone.vgg.vgg19_bn",
    "resnet": "model.backbone.resnet.ResNet",
    "resnet18": "model.backbone.resnet.resnet18",
    "resnet34": "model.backbone.resnet.resnet34",
    "resnet50": "model.backbone.resnet.resnet50",
    "resnet101": "model.backbone.resnet.resnet101",
    "resnet152": "model.backbone.resnet.resnet152",
    "densenet121": "model.backbone.densenet.densenet121",
    "densenet161": "model.backbone.densenet.densenet161",
    "densenet169": "model.backbone.densenet.densenet169",
    "densenet201": "model.backbone.densenet.densenet201",
    "vit": "model.backbone.vision_transformer.VisionTransformer",
    "vit_b_16": "model.backbone.vision_transformer.vit_b_16",
    "vit_b_32": "model.backbone.vision_transformer.vit_b_32",
    "vit_l_16": "model.backbone.vision_transformer.vit_l_16",
    "vit_l_32": "model.backbone.vision_transformer.vit_l_32",
    "vgg11ae": "model.backbone.simple_autoenc.vgg11ae",
    "bisenetv2_mbnt_semantic": "model.backbone.bisenetv2.BiSeNetv2_mbnt_semantic",
    "bisenetv2_resnet_semantic": "model.backbone.bisenetv2.BiSeNetv2_resnet_semantic",
    "mae": "model.backbone.mae.MAE",
    # fuser
    "simple_concat_fuser": "model.fuser.concat.Simple_concat_fuser",
    "map_vector_stacker": "model.fuser.concat.Map_vector_stacker",
    "bisenet_bga": "model.fuser.bisenet_bga.BGALayer",
    # head
    "mlp": "model.head.mlp.MLP",
    "seg": "model.head.seg.SEG",
    "ssl_simsiam": "model.head.ssl_simsiam.SSL_SimSiam",
    "ssl_supcon": "model.head.ssl_supcon.SSL_CONTRASTIVE",
    "recon_decoder": "model.head.recon_decoder.resnet_decoder_all",
    "identity": "model.head.identity.Identity",
    "regressor": "model.head.regressor.Regressor"
}



# designed to wrap the module, and format the input with args
class Module_wrapper(Module_base):
    def __init__(self, module_class, name: str, input_shape, **kwargs):
        Module_base.__init__(self, name, input_shape)
        self.module = module_class(**kwargs)
        # TODO: add support for Zheng's functions
        # some functions return class (vgg.py); some return instances ()
        self.add_module("module", self.module)

    # note that it is assumed that X is a list of tensor, already prepared for the content
    # so we should only need to spread it
    def forward(self, X):
        return self.module(*X)

    def __call__(self, X, **kwargs):
        return self.module(*X)


def get_module(module_name, **kwargs):
    if module_name in _private_imports:

        module_path, class_name = _private_imports[module_name].rsplit(".", 1)
        module = importlib.import_module(module_path)
        module_class = getattr(module, class_name)
    
        # module_class = _modules[module_name]
        # note we assume module_class is either a class or function, 
        # if it is a function, it is assumed that it returns nn.Module, so we mandatory wrap it
        if isinstance(module_class, type) and issubclass(module_class, Module_base):
            module = module_class(**kwargs)
            return module
        return Module_wrapper(module_class, **kwargs)
    else:
        raise Exception(Color_string.fail("[module_loader]Module not found:%s" % module_name))

if __name__=='__main__':
    print(get_module("mlp"))
    print(get_module("mobilenet_v2"))
    print(get_module("mobilenet_v1"))
