"""
Supported modulus training paradigms
"""

import torch

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, II

from .loss import NTKConf


@dataclass
class TrainingConf:
    max_steps: int = MISSING
    grad_agg_freq: int = MISSING
    rec_results_freq: int = MISSING
    rec_validation_freq: int = MISSING
    rec_inference_freq: int = MISSING
    rec_monitor_freq: int = MISSING
    rec_constraint_freq: int = MISSING
    save_network_freq: int = MISSING
    print_stats_freq: int = MISSING
    summary_freq: int = MISSING
    amp: bool = MISSING
    amp_dtype: str = MISSING


@dataclass
class DefaultTraining(TrainingConf):
    max_steps: int = 10000
    grad_agg_freq: int = 1
    rec_results_freq: int = 1000
    rec_validation_freq: int = II("training.rec_results_freq")
    rec_inference_freq: int = II("training.rec_results_freq")
    rec_monitor_freq: int = II("training.rec_results_freq")
    rec_constraint_freq: int = II("training.rec_results_freq")
    save_network_freq: int = 1000
    print_stats_freq: int = 100
    summary_freq: int = 1000
    amp: bool = False
    amp_dtype: str = "float16"

    ntk: NTKConf = NTKConf()


@dataclass
class VariationalTraining(DefaultTraining):
    test_function: str = MISSING
    use_quadratures: bool = False


@dataclass
class VariationalTraining(DefaultTraining):
    test_function: str = MISSING
    use_quadratures: bool = False


def register_training_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="training",
        name="default_training",
        node=DefaultTraining,
    )

    cs.store(
        group="training",
        name="variational_training",
        node=VariationalTraining,
    )
