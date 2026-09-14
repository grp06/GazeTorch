# 2. `data` Section
by Zhenhao.

`data` Section guides GazeTorch to construct `torch.utils.data.Dataset` instances later used in training and validation.
It should contains nested `dict`s, `dataset_train` and/or `dataset_val`.
Either one contains:
* `type:str`: the dataset name, which should be registered in `data.dataset_loader`. GazeTorch will invoke `get_dataset()` method and generate an `torch.utils.data.Dataset` instance accordingly.
* `args:dict`: the arguments passed to the `Dataset` class constructor.