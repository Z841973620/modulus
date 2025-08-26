import torch
import torch.distributed as dist

import logging
import os
import time
import numpy as np

logger = logging.getLogger("__name__")

# Create singleton DistributedManager class
class DistributedManager(object):
    _shared_state = {}

    def __new__(cls):
        obj = super(DistributedManager, cls).__new__(cls)
        obj.__dict__ = cls._shared_state

        # Set the defaults
        if not hasattr(obj, "_rank"):
            obj._rank = 0
        if not hasattr(obj, "_world_size"):
            obj._world_size = 1
        if not hasattr(obj, "_local_rank"):
            obj._local_rank = 0
        if not hasattr(obj, "_distributed"):
            obj._distributed = False
        if not hasattr(obj, "_device"):
            obj._device = torch.device(
                f"cuda:0" if torch.cuda.is_available() else "cpu"
            )
        if not hasattr(obj, "_cuda"):
            obj._cuda = torch.cuda.is_available()
        if not hasattr(obj, "_broadcast_buffers"):
            obj._broadcast_buffers = False
        if not hasattr(obj, "_find_unused_parameters"):
            obj._find_unused_parameters = False
        if not hasattr(obj, "_cuda_graphs"):
            obj._cuda_graphs = False

        return obj

    @property
    def rank(self):
        return self._rank

    @property
    def local_rank(self):
        return self._local_rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def device(self):
        return self._device

    @property
    def distributed(self):
        return self._distributed

    @property
    def cuda(self):
        return self._cuda

    @property
    def broadcast_buffers(self):
        return self._broadcast_buffers

    @broadcast_buffers.setter
    def broadcast_buffers(self, broadcast: bool):
        self._broadcast_buffers = broadcast

    @property
    def find_unused_parameters(self):
        return self._find_unused_parameters

    @find_unused_parameters.setter
    def find_unused_parameters(self, find_params: bool):
        if find_params:
            # Logger may not be config'd here yet
            logger.warning(
                "Setting `find_unused_parameters` in DDP to true, use only if necessary."
            )
        self._find_unused_parameters = find_params

    @property
    def cuda_graphs(self):
        return self._cuda_graphs

    @cuda_graphs.setter
    def cuda_graphs(self, graphs: bool):
        # Function for any modifications needed for DDP using cuda graphs
        if graphs and self._find_unused_parameters:
            # Logger may not be config'd here yet
            logger.warning(
                "DDP `find_unused_parameters` must be false for CUDA graphs."
            )
            raise ValueError(
                "`cuda_graphs` and `find_unused_parameters` cannot both be true"
            )

        self._cuda_graphs = graphs

    @staticmethod
    def get_available_backend():
        if torch.cuda.is_available() and torch.distributed.is_nccl_available():
            return "nccl"
        else:
            return "gloo"

    @staticmethod
    def initialize_env():
        rank = int(os.environ.get("RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ.get("LOCAL_RANK"))
        else:
            local_rank = rank % torch.cuda.device_count()
        addr = os.environ.get("MASTER_ADDR")
        port = os.environ.get("MASTER_PORT")

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
        )

    @staticmethod
    def initialize_open_mpi(addr, port):
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK"))
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE"))
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
            method="openmpi",
        )

    @staticmethod
    def initialize_slurm(port):
        rank = int(os.environ.get("SLURM_PROCID"))
        world_size = int(os.environ.get("SLURM_NPROCS"))
        local_rank = int(os.environ.get("SLURM_LOCALID"))
        addr = os.environ.get("SLURM_LAUNCH_NODE_IPADDR")

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
            method="slurm",
        )

    @staticmethod
    def initialize():
        addr = os.getenv("MASTER_ADDR", "localhost")
        port = os.getenv("MASTER_PORT", "12355")
        # https://pytorch.org/docs/master/notes/cuda.html#id5
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        try:
            DistributedManager.initialize_env()
        except:
            if "SLURM_PROCID" in os.environ:
                DistributedManager.initialize_slurm(port)
            elif "OMPI_COMM_WORLD_RANK" in os.environ:
                DistributedManager.initialize_open_mpi(addr, port)

        # Set per rank numpy random seed for data sampling
        np.random.seed(seed=DistributedManager().rank)

        manager = DistributedManager()
        if manager.distributed:
            print(
                f'Initialized process {manager.rank} of {manager.world_size} using method "{manager._initialization_method}". Device set to {str(manager.device)}'
            )

    @staticmethod
    def setup(
        rank=0,
        world_size=1,
        local_rank=None,
        addr="localhost",
        port="12355",
        backend="nccl",
        method="env",
    ):
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)

        manager = DistributedManager()

        manager._distributed = (world_size > 1) and torch.distributed.is_available()
        if manager._distributed:
            # Update rank and world_size if using distributed
            manager._rank = rank
            manager._world_size = world_size
            if local_rank is None:
                manager._local_rank = rank % torch.cuda.device_count()
            else:
                manager._local_rank = local_rank

            # Setup distributed process group
            # time.sleep(1)
            dist.init_process_group(
                backend, rank=manager.rank, world_size=manager.world_size
            )

        manager._device = torch.device(
            f"cuda:{manager.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        # Needed for cuda graphs
        if torch.cuda.is_available():
            torch.cuda.set_device(manager.local_rank)

        manager._initialization_method = method

        # Set device for this process and empty cache to optimize memory usage
        torch.cuda.device(manager.device)
        torch.cuda.empty_cache()

    @staticmethod
    def cleanup():
        dist.destroy_process_group()
