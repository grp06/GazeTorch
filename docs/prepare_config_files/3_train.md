# 3. `train` Section
by Justin Cai and Zhenhao.

`train` Sections specifies runtime settings for training.

## General training hyperparameters
* `num_epochs: int`: total # of epochs
* `seed: int=0`: seed # for random number generators

## GPU settings
* `gpus: list[int]`: which GPUs to use
  * Note: `CUDA_VISIBLE_DEVICES="..."` in runtime will affect GPU ordering.
* `batch_per_gpu: int`: minibatch size per GPU

## Checkpoint
Note: this is different from `net.ckpt`, which specifies that starting ckpt for initialization.
`train.ckpt_dir` here specifies the target ckpt path to save models.

* `ckpt_dir: str`: target ckpt folder (where ckpt & Tensorboard logs are kept).
* `ckpt_period: int`: save a ckpt every ~ epochs.
* `ckpt_auto_clear: bool=True`: before training, whether to clear the ckpt if it's not empty
    * Remember to switch off this feature if you want to resume training at `ckpt` dir!
* `reset_epoch: bool=True`: if True, always start from Epoch #0. Should be `False` if resuming a training task from checkpoint.

## Loss
`loss: dict` Subsection specifies the loss setting.
Users need to define the **loss terms** first, and then define `loss_train` and `loss_val`, both of which are linear combinations of the loss terms. Example:

```YAML
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
      loss1: 0.5
      loss2: 1
```

Each loss term is specified by a `dict` with a specific name, similar to a module in `net` Section. It contains:
* `module_input: OrderedDict`: each item specifies an input Tensor fed to the loss function.
    * Definition is similar to `net.<any module>.module_input`.
    * There is one reserved key, `dataset`, indicating the Tensor is a ground truth label given by the dataset. The Tensor list of dataset follows the `labels` argument when creating the dataset (see `data` Section).
* `type: str`: loss type, which should be registered in `train.loss_loader`. GazeTorch will invoke `get_loss()` method and generate a `torch.nn.modules.loss._Loss` instance accordingly.
* `args: dict`: *(upcoming feature)* other arguments to be passed to the loss function on invocation (namely `**kwargs`).

And there are two mandatory terms, `loss_train` and `loss_val`, specifying the loss function definition in training and validation phase. Either is a `dict` where each item goes like "`loss term name`: `weight`". In the example above, `loss_val` = `loss1`*0.5 + `loss2`.

## Weight initialization
At present this is under `net` Section - please refer to ckpt settings there.

## Optimizer
`Optimizer:dict` Subsection specifies the optimizers to be used in training phase (to perform back-propagation).
It should contain at least one global optimizer called `default`. User can define more optimizers if they want different BP behaviour (e.g. different learning rate) on certain modules.
Example:

```YAML
  optimizer:
    default:
      type: adam
      lr: 3e-4
      lr_scheduler: # use StepLR
        type: step
        args:
          step_size: 5
          gamma: 0.9
      args:

    mlp:
      type: adamw
      lr: 1e-4
      args:
```
Each optimizer (including `default`) definition should contain
* `type:str`: optimizer type, which should be registered in `train.optimizer_loader`. GazeTorch will invoke `get_optimizer()` method and generate an `torch.optim.Optimizer` instance accordingly.
* `lr: float`: learning rate.
* `args: dict`: other arguments to be passed to the optimizer constructor (namely `**kwargs`).
* `lr_scheduler: dict`: optional. If specified, then create a `torch.optim.lr_scheduler._LRScheduler` instance corresponding to the optimizer. See details in `LR_Scheduler` section.

## LR_Scheduler (LRS)
An LRS adjusts the learning rate per optimizer for each epoch. If not specified at all, GazeTorch will apply fixed learning rate(s) for all epochs.

Like optimizers, Users can either specify a global `default` LRS by putting it under `train.optimizer.default` section, or specify an LRS just working for one optimizer by putting it under the corresponding optimizer section.

Each LRS defintion should contain
* `type:str`: LRS type, which should be registered in `train.optimizer_loader`. GazeTorch will invoke `get_lr_scheduler()` method and generate an `torch.optim.lr_scheduler._LRScheduler` instance accordingly.
* `args:dict`: other arguments to be passed to the optimizer constructor (namely `**kwargs`).

Notes:
* If a non-default optimizer definition has no LRS section, then this optimizer will be paired with a default LRS (if specified).

## Multi-view settings

Some loss terms (e.g. contrastive loss) require multiple "views" of each sample.
* `num_views: int=1`: view number in training phase.
* `flip_every_other_view: bool=False`: if True, make view #(even) = not horizontally flipped and view #(odd) = flipped.
* `num_views_val: int=1`: view number in validation phase.
* For `hybrid_v2` datasets, these features are moved to `data.dataset_train/val`.

## Validation phase
* `val_period: int=1`: perform validation per ~ epochs. 
* `no_val_before: int=0`: skip validation phase until epoch No.~.

## Runtime, console and debug settings
* `infer_test: int=0`: print ~ mini-batches' output after each training epoch. Skipped if set to `0`.
* `iters_limit: int=0`: if not `0`, only first ~ iterations will be run per epoch. Make each epoch shorter for faster debugging (similar to debug mode in datasets - use either one).
* `log_period: int`: print loss per ~ iterations on console. Print nothing if set to `0`.