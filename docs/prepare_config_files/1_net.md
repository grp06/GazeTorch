# 1. `net` Section
by Justin Cai and Zhenhao.


`net` Section specifies how the entire Net should look like before training begins. Task includes identifying output/input format, initialize child modules, connecting them correctly, and lastly load in ckpt if instructed

(Note, if you find it hard to understand each field, try referring to [`sample.yaml`](../sample.yaml))

## Terminologies
### **model**
A model refers to the whole net you are constructing, that is everything you defined under net of your sample_config.yaml. In most situations though, it is refered to as the neural network architecture.

### **module**
A module refers to the sub-component of a net, it is one of the following: backbone, fuser, head. Note that **loss** is very similar to a module, but for the purpose of training we make it a different concept of its own

## Parameters at the root

Every config file's `net` Section should begin with the following attributes:

* `output: list[str]`: defines which submodule(s) provide the output of the entire model. It is a list containing the **names** of output submodules.
    * example: `[head]`
* `input: list[list[int]]`: defines the expected input. It should be an instance of `list[list[int]]`, a list of tensors **by their dimensions (without the batch dimension)**.
    * example: `[[3, 224, 224], [2]]` (taking 2 Tensors as input)
* `ckpt: str`: (optional) the starting ckpt for the entire Net. If given, GazeTorch will train starting from this ckpt; otherwise GazeTorch will train from scratch.
It contains `dir` which indicate the directory where I can find the ckpts, and `name` which indicate which ckpt I should load. `name` is optional, when not provided, it takes the latest ckpt.
    * example:
    ```YAML
    ckpt:
        dir: /root/GazeTorch_test/GazeTorch/ckpts
        name: ckpt_epoch_00000.pyth
    ```

## Constituent Modules
The remaining of `Net` Sections are unlimited amount of modules that compose the model. These modules are required to be defined following a specific format, so once `net.py` picks it up it will be able to construct the model correctly.
Here is an example:

```YAML
face_feature:
    module_input: # OrderedDict
      input: all
    module_output: [[512]] # list[list[int]]
    type: resnet50
    args:
      num_classes: 512
    # module_ckpt:
    #   ckpt_module_dir: /root/GazeTorch_test/GazeTorch/playground/2
    #   ckpt_module_name: ckpt_epoch_00001.pyth

head:
    module_input:
      face_feature: all
    module_output: [[2]]
    type: mlp
    # module_ckpt:s
    #   ckpt_module_dir: /root/GazeTorch_test/GazeTorch/playground/2
    #   ckpt_module_name: ckpt_epoch_00001.pyth
    args:
      channels: [64, 2]
```

This module's `name` is **face_feature**. Under the name, user needs to define:
* `module_input: OrderedDict`: specifies the input Tensor(s) of this module. Each item corresponds to one or a group of Tensor, where
    * key = 
        * `input`, if the input Tensor(s) are just from input Tensor(s), or
        * a module's name, if the input Tensor(s) are form this module's output.
    * value = which of the Tensors from the `key`. It can be
        * `all`: taking all Tensors
        * an `int`: taking the Tensor at this position.
            * If you want to take multiple Tensors from a module "A", put multiple items with key="A", enumerating the indices.
    
    In runtime, all items will be concatenated into a single array of Tensors following the item order.
* `module_output: list[list[int]]`: lists the dimensions of each output Tensor.

    Notes: just for better readibility - not actually taking effect in runtime - 
the output Tensor dimensions are given by the corresponding `Module_base` instance.

* `type: str`: specifies the module type, which should be registered in `model.module_loader`. GazeTorch will invoke `get_module()` method and generate a `Module_base` instance accordingly.
* `args: dict`: the argument list fed into `get_module()` (namely the `**kwargs` / which is later passed to the constructor of the module class).

* `module_ckpt`: (optional) exactly same format as `ckpt` at root.
Use this when you want the module initialized by a different ckpt form the root ckpt.
In runtime, GazeTorch will load root ckpt first and then overwrite the modules wherever `module_ckpt` is specified.