# Config Loader
The goal of the config reader is to create a simple to access, simple to debug, simple to update modules which focuses on reading a config file in yaml format into a fully functional object. It has some debug messages, print itself quiet nicely using the build in python print function, and access it at ease.

# Current Architecture
We have 3 seperate areas of config related to our project:
## net
this part of the config contains information relevant to the general structure of the net, we treat every large transition between network as module (blackbox) and hook them up using the mandatory attribute shared between them, here is an example:
```YAML
  # component in red
  fuser:
    input_source: 
      face_backbone: all
      left_eye_backbone: all
      right_eye_backbone: all
    output_format: [3, null, 64] 
    type: multi_model_bottleneck
    args:
      is_training: True
      layer_num: 6
      num_heads: 16
      token_dim: 64
      dropout: 0.1
      ff_hidden_channel_ratio: 2
      ff_hidden_activation: relu
      type: Cross
      recursive:
        inner1: true
        inner2: true
```
(to help you understand better, here is a general diagram of the trident network: Trident sample flow.pdf)

within this component, the mandatory components are:
| name        | Usage           |
| ------------- |:-------------:| 
| input_source    | indicating which previous module does it extract input from |
| output_format     | used to confirm and validate that the output format of the current module, useful when debugging and logging   |
| type | indicate the type of module, thus module user can correctly identify how to initialize the module|
| args | used to contain any other parameter needed to initialize the module, it is usually type-dependent|

Other than that, net also has some basic support parameter

```YAML
  output: head
  input: [[224,224,3], [74,74,3], [74,74,3]]
```

while output_node help us identify which module should net used as output, input indicate the dimension and the amount of tensor which our network will accept as input

## train
train contains any neccessary information needed to run the trainning code, it recursively reads any parameter. here is an example:
```YAML
train:
  gpus: [4,5,6,7]
  gpu_base: 7
  batch_per_gpu: 32 # normal is 64
  # sample ckpt storeage control
  ckpt_dir: /root/GazeTorch_test/GazeTorch/playground/
  ckpt_period: 1
  ckpt_auto_clear: True
  num_epochs: 5
  log_period: 10
  val_period: 1
  infer_test: False # for debug
  iters_limit: 10 # for debug
  seed: 0

  optimizer:
    face_feature:
      type: adam
      lr: 3e-4
      args:
    default:
      type: adam
      lr: 3e-5
      args:
  
  loss:
    loss1: #l(x, gt) L([Tensor])
      module_input:
        head: all # from Net
        dataset: 0 # from Dataset
      type: l1
    loss2:
      module_input:
        head: all
        dataset: 0 # from Dataset
      type: angular

    loss_train:
      loss1: 1

    loss_val:
      loss2: 1
  
```
(0 restriction is placed in train)

## data
Contains any data related information, it basically unlimited amount of dataset, here is an example:

```YAML
  dataset_train:
    type: Dataset_MPIIGAZE
    args:
      mean: [0.5,0.5,0.5]
      std: [0.2,0.2,0.2]
      aug: 10
      filter:
        person_id: 0
        # sample_per_person: 10000
        inclusive: False
```

the only mandatory parameter is just type and args, similar usage as to net.
(might be updated in the future to introduce more utilities)


# How to use
Simply do import, here is an example:
```python
from config.config_loader.config_loader import Config_loader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="cfg")
args = parser.parse_args()

config_loader = Config_loader(args.cfg)
print(config_loader)
```

to access, simply do something like the following;
```python
print(self.net.face_backbone.input_source.face)
```
the print is also beautified, here is a quick example of print
```YAML
        dataset_vis: 
                type: Dataset_MPIIGAZE
                args: 
                        filter: 
                                person_id: 0
                                inclusive: False
                        mean: [0.5, 0.5, 0.5]
                        std: [0.2, 0.2, 0.2]
                        aug: 10
```
(the actual print also have color, so if you want to see an example, go test it!)

## How to Update
Start with config_loader.py and dig down, code should be relatively simple to read
you can easily add new feature or enforce new rules to the existing parsing. (for example, how to recursively parse config.yaml)