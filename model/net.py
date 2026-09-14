import torch
import torch.nn as nn
from collections import defaultdict

from config.config_loader.config_base import Config_base
from config.config_loader.config_net import Config_net_blackbox, Config_net_ckpt
from model.module_loader import get_module
from tools.printer import Printer, Color_string
from tools.checkpoint import Checkpoint_io
from tools.param_constant import *


class _Model(nn.Module):
    def __init__(self, config, topologically_sorted_modules):
        super(_Model, self).__init__()
        self.topologically_sorted_modules = topologically_sorted_modules
        self.config = config
        self.initialized_modules = nn.ModuleDict()
        for module_name in self.topologically_sorted_modules:
            if module_name != INPUT:
                module_config = getattr(config, module_name)  # the general module config

                module_type = module_config.type  # extract the type
                input_shape = []  # calculate input shape based on input sources
                for input_name, split in module_config.module_input.__dict__.items():
                    # note that no matter what we receive, we always flatten it into a 2d array
                    if input_name != INPUT:  # as long as it is not the initial inputs
                        format = getattr(config, input_name).module_output
                        if split == ALL:
                            input_shape += format
                        else:
                            input_shape += [format[split]]
                    else:
                        input_shape += config.input
                args = module_config.args.__dict__ if module_config.has(ARGS) else {}
                self.initialized_modules[module_name] = get_module(
                    module_name=module_type,
                    **args,  # spread extra args into the components
                    # rest are mandatory parameters for Module_base class
                    name=module_name,
                    input_shape=input_shape,
                    # output_shape=module_config.output_format, (LEGACY)
                )

    def forward(self, X):
        """
        X: list[torch.Tensor] following config.input
        during forwarding, we forward in order of topologically_sorted_modules, store result in
        self.computed_module, and keep going until every module is traversed
        """
        self.computed_module = {INPUT: X}
        for module_name in self.topologically_sorted_modules:
            if module_name != INPUT:
                # first identify the input source
                input = []
                for source, split in getattr(self.config, module_name).module_input.__dict__.items():
                    if split == ALL:  # we don't care about the dimesion
                        for tensor in self.computed_module[source]:
                            input.append(tensor)
                    else:
                        input.append(self.computed_module[source][split])
                self.computed_module[module_name] = self.get_module(module_name)(input)
                if torch.is_tensor(self.computed_module[module_name]):  # convert single tensor to a list containing it
                    self.computed_module[module_name] = [self.computed_module[module_name]]
        # if there are mutiple output required, sort them into list
        if isinstance(self.config.output, list):
            results = {}
            for out in self.config.output:
                results[out] = self.computed_module[out]
            return results
        else:
            raise("[Net]: output must be a list")

    def get_module(self, module_name):
        return self.initialized_modules[module_name]

    def get_all_modules(self):
        return self.initialized_modules

class Net_loader:
    def __init__(self, config) -> None:
        self.input = config.input
        self.output = config.output
        self.config = config

        self.modules = [
            (name, module) for name, module in config.__dict__.items() if isinstance(module, Config_net_blackbox)
        ]  # extract all Config_base
        # use topological sort to generate the execution graph of module
        self.topologically_sorted_modules = self.topologial_sort()
        # Now the topological sort is completed, we now initialize the component one by one
        # self.initialized_modules = self.initialize_modules(config)
        # self.initialize_modules(config)
        self.model = _Model(config, self.topologically_sorted_modules)

        self.start_epoch = 0
        self.load_from_ckpt()
        Printer.print_OK("[Net]Net loaded & initialized.")

    def forward(self, X):
        """
        X: list[torch.Tensor] following config.input
        during forwarding, we forward in order of topologically_sorted_modules, store result in
        self.computed_module, and keep going until every module is traversed

        returns a dict. (see _Model.forward)
        """
        return self.model(X)

    def infer(self, X0):
        """
        X: list[torch.Tensor] following config.input
        during forwarding, we forward in order of topologically_sorted_modules, store result in
        self.computed_module, and keep going until every module is traversed

        The only difference from forward() is X may be not batched. So we make X into a 1-batch where each element is 4d Tensor(NCHW).
        """
        X = X0.copy()
        for i, x in enumerate(X):
            x = torch.Tensor(x)
            if x.ndim<4:
                X[i] = x[None, ...]
        return self.forward(X)

    def topologial_sort(self):
        """
        every network is a Directed Acyclic Graph, A.K.A a directed graph with no cycle.
        that means we can use topological sort to find a possible scheduling of the net,
        such that each module has it pre-req satisfied in order
        """

        # the graph
        self.graph = defaultdict(list)

        for name, module in self.modules:
            if not isinstance(module.module_input, Config_base):
                raise Exception(
                    Color_string.fail("illegal argument, seems like one of the module_input is not defined properly")
                )
            # note that split can only be one of the following: [All, 0, 1 ....., source.shape[0] - 1]
            # however, split is not used here, it will be used later to validate the graph once the graph is complete
            for source, split in module.module_input.__dict__.items():
                self.graph[source].append(name)
            if not self.graph[name]:  # if this is one of the module with out-degree = 0, we initialize it to empty
                self.graph[name] = []

        # we now have formed a directed graph of net, run topoologial sort

        indegree = {node: 0 for node in self.graph}
        for node, edges in self.graph.items():
            for edge in edges:
                indegree[edge] += 1

        # the default assumption is that "input" has in degree of 0, so we check

        if indegree[INPUT] != 0:
            raise Exception(Color_string.fail("input does not have in degree of 0, please check cfg"))

        nodes_completed = [INPUT]
        # container for result
        topologically_sorted_modules = []

        while len(nodes_completed) > 0:
            node = nodes_completed.pop()
            topologically_sorted_modules.append(node)

            for neigh in self.graph[node]:
                indegree[neigh] -= 1
                if indegree[neigh] == 0:
                    nodes_completed.append(neigh)

        if len(topologically_sorted_modules) != len(self.graph):
            raise Exception(Color_string.fail("Graph generated by net config is not a Directed Acyclic Graph"))

        return topologically_sorted_modules

    def get_module(self, module_name):
        return self.model.get_module(module_name)

    def get_all_modules(self):
        return self.model.get_all_modules()

    def load_from_ckpt(self):
        cfg_ckpt = self.config.get(CKPT, None)
        
        # we might hit the case where only partial loading is needed
        ckpt_dir = cfg_ckpt.dir if cfg_ckpt else None # "dir" always exists if cfg_ckpt is not none
        ckpt_name = cfg_ckpt.get(NAME, None) if cfg_ckpt else None 
        overriding_ckpts = {}
        for name, module in self.modules:
            if module.has(CKPT_MODULE):
                overriding_ckpts[name] = module.ckpt_module
        cio = Checkpoint_io(ckpt_dir)
        if cio.has_ckpt():
            epoch = cio.load_ckpt(
                ckpt_name, # if ckpt_start_name is not specified then take the last available ckpt
                self.model,
                overriding_ckpts
            )
            Printer.print_OK("[Net]Ckpt loaded from: {}".format(ckpt_dir))
        else:
            epoch = 0
        self.start_epoch = epoch
        Printer.print_OK("[Net]Current epoch = {}".format(epoch))
    

