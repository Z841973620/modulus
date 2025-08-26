"""
Supported Modulus graph configs
"""

import torch

from dataclasses import dataclass
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore
from hydra.experimental.callback import Callback
from typing import Any
from omegaconf import DictConfig, OmegaConf

from modulus.distributed import DistributedManager


class ModulusCallback(Callback):
    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        # Update dist manager singleton with config parameters
        manager = DistributedManager()
        manager.broadcast_buffers = config.broadcast_buffers
        manager.find_unused_parameters = config.find_unused_parameters
        manager.cuda_graphs = config.cuda_graphs


DefaultCallbackConfigs = DictConfig(
    {
        "modulus_callback": OmegaConf.create(
            {
                "_target_": "modulus.hydra.callbacks.ModulusCallback",
            }
        )
    }
)


def register_callbacks_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="hydra/callbacks",
        name="default_callback",
        node=DefaultCallbackConfigs,
    )
