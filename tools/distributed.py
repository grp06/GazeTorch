import torch
import torch.distributed as dist

import sys
import builtins
import logging

class Distributed_utils:
    @staticmethod
    def is_master_proc():
        """
        Determines if the current process is the master process.
        """
        if torch.distributed.is_initialized():
            return dist.get_rank()  == 0
        else:
            return True

    @staticmethod
    def get_world_size():
        """
        Get the size of the world.
        """
        if not dist.is_available():
            return 1
        if not dist.is_initialized():
            return 1
        return dist.get_world_size()

    @staticmethod
    def get_rank():
        """
        Get the rank of the current process.
        """
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    @staticmethod
    def _suppress_print():
        """
        Suppresses printing from the current process.
        """

        def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
            pass

        builtins.print = print_pass

    @staticmethod
    def setup_logging():
        """
        Sets up the logging for multiple processes. Only enable the logging for the
        master process, and suppress logging for the non-master processes.
        """
        # Set up logging format.
        _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

        if Distributed_utils.is_master_proc():
            # Enable logging for the master process.
            logging.root.handlers = []
            logging.basicConfig(
                level=logging.INFO, format=_FORMAT, stream=sys.stdout
            )
        else:
            # Suppress logging for non-master processes.
            Distributed_utils._suppress_print()

    @staticmethod
    def generate_order_list(n, parallel=False, group_size=1):
        '''
        Generate a random permutation of length `n`.
        If parallel, let the main process do the generation and broadcast it to other processes.

        group_size: keep every block of `group_size` consecutive indices together
                    (shuffle GROUPS, not individual indices). Needed when the dataset
                    lays out n_views of a sample contiguously (Dataset_Hybrid_V2): a
                    plain per-index shuffle would scatter a sample's views apart.
                    group_size=1 -> original per-index behaviour.
        '''
        import numpy as np
        assert n % group_size == 0, \
            f"generate_order_list: n={n} not divisible by group_size={group_size}"

        def make():
            groups = np.random.permutation(n // group_size)            # shuffle GROUPS
            # expand each group id into its `group_size` consecutive positions
            return (groups[:, None] * group_size + np.arange(group_size)).reshape(-1).astype(int)

        if parallel:
            if Distributed_utils.is_master_proc():
                tensor = torch.from_numpy(make()).cuda()
            else:
                tensor = torch.empty(n, dtype=int).cuda()

            dist.broadcast(tensor, src=0)
            return tensor.tolist()
        else:
            return make()