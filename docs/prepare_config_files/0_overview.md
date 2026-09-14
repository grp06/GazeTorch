# Guide: Prepare a YAML config file for GazeTorch training

by Justin Cai and Zhenhao.

## 0. Overview

### How does a config file work?

So instead of some weird parameter that we will force you to put in, we simply merged absolutely everything into a single YAML config file which will help us define the system. As for reference, please checkout `sample.yaml` - a config file that is ready to go, and will show how to train a simple network with some useful comment.

In particular, there are 3 sections:

* `net`
* `data`
* `train`

Wrap all your demands about the task into a config file by filling these sections, then feed it into `gazetorch.py`, and GazeTorch should be able to train models as you wish.

### General rules

* All config files should be in  `.yaml` format.
* Use indents properly.
* Use `#` for comments.

## To load a config file
Use `config.config_loader.Config_loader` to load a full `.yaml` file (which must contain `net`, `train` and `data` sections), which returns a `Config_base` instance.

* To load a minimalistic config file (for example, if you want to test `data` section only), use `config.config_loader.load_yaml()` method instead, which returns a `CfgNode` instance.
* The usage of `Config_base` and `CfgNode` are all the same. 
The only difference is, `Config_base` has an internal parser to examine if the necessary fields are present in the config file.

