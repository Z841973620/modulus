"""
Architecture/Model configs
"""

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI, II
from typing import Any, Union, List, Dict, Tuple


@dataclass
class ModelConf:
    _target_: str = MISSING


@dataclass
class FullyConnectedConf(ModelConf):

    _target_: str = "modulus.models.fully_connected.FullyConnectedArch"
    layer_size: int = 512
    nr_layers: int = 6
    skip_connections: bool = False
    # activation_fn = layers.Activation.SILU
    adaptive_activations: bool = False
    weight_norm: bool = True


@dataclass
class FusedMLPConf(ModelConf):
    _target_: str = "modulus.models.fused_mlp.FusedMLPArch"
    layer_size: int = 128
    nr_layers: int = 6
    # activation_fn = layers.Activation.SIGMOID


@dataclass
class FusedFourierNetConf(ModelConf):
    _target_: str = "modulus.models.fused_mlp.FusedFourierNetArch"
    layer_size: int = 128
    nr_layers: int = 6
    # activation_fn = layers.Activation.SIGMOID
    n_frequencies: int = 12


@dataclass
class FusedGridEncodingNetConf(ModelConf):

    _target_: str = "modulus.models.fused_mlp.FusedGridEncodingNetArch"
    layer_size: int = 128
    nr_layers: int = 6
    # activation_fn = layers.Activation.SIGMOID
    indexing: str = "Hash"
    n_levels: int = 16
    n_features_per_level: int = 2
    log2_hashmap_size: int = 19
    base_resolution: int = 16
    per_level_scale: float = 2.0
    interpolation: str = "Smoothstep"


@dataclass
class FourierConf(ModelConf):
    _target_: str = "modulus.models.fourier_net.FourierNetArch"
    # frequencies=("axis", [i for i in range(10)]),
    # frequencies_params=("axis", [i for i in range(10)]),
    # activation_fn=layers.Activation.SILU,
    layer_size: int = 512
    nr_layers: int = 6
    skip_connections: bool = False
    weight_norm: bool = True
    adaptive_activations: bool = False


@dataclass
class HighwayFourierConf(ModelConf):
    _target_: str = "modulus.models.highway_fourier_net.HighwayFourierNetArch"
    # frequencies: Any = ("axis", [i for i in range(10)])
    # frequencies_params: Any = ("axis", [i for i in range(10)])
    # activation_fn=layers.Activation.SILU
    layer_size: int = 512
    nr_layers: int = 6
    skip_connections: bool = False
    weight_norm: bool = True
    adaptive_activations: bool = False
    transform_fourier_features: bool = True
    project_fourier_features: bool = False


@dataclass
class ModifiedFourierConf(ModelConf):
    _target_: str = "modulus.models.modified_fourier_net.ModifiedFourierNetArch"
    # frequencies: Any = ("axis", [i for i in range(10)])
    # frequencies_params: Any = ("axis", [i for i in range(10)])
    # activation_fn=layers.Activation.SILU
    layer_size: int = 512
    nr_layers: int = 6
    skip_connections: bool = False
    weight_norm: bool = True
    adaptive_activations: bool = False


@dataclass
class MultiplicativeFilterConf(ModelConf):
    _target_: str = (
        "modulus.models.multiplicative_filter_net.MultiplicativeFilterNetArch"
    )
    layer_size: int = 512
    nr_layers: int = 6
    skip_connections: bool = False
    # activation_fn=layers.Activation.IDENTITY
    # filter_type: FilterType = FilterType.FOURIER
    weight_norm: bool = True
    input_scale: float = 10.0
    gabor_alpha: float = 6.0
    gabor_beta: float = 1.0
    normalization: Any = (
        None  # Change to Union[None, Dict[str, Tuple[float, float]]] when supported
    )


@dataclass
class MultiscaleFourierConf(ModelConf):
    _target_: str = "modulus.models.multiscale_fourier_net.MultiscaleFourierNetArch"
    # frequencies: Any = field(default_factory=lambda: [32])
    frequencies_params: Any = None
    # activation_fn=layers.Activation.SILU,
    layer_size: int = 512
    nr_layers: int = 6
    skip_connections: bool = False
    weight_norm: bool = True
    adaptive_activations: bool = False


@dataclass
class SirenConf(ModelConf):
    _target_: str = "modulus.models.siren.SirenArch"
    layer_size: int = 512
    nr_layers: int = 6
    first_omega: float = 30.0
    omega: float = 30.0
    normalization: Any = (
        None  # Change to Union[None, Dict[str, Tuple[float, float]]] when supported
    )


@dataclass
class MultiresolutionHashNetConf(ModelConf):
    _target_: str = "modulus.models.hash_encoding_net.MultiresolutionHashNetArch"
    layer_size: int = 64
    nr_layers: int = 3
    skip_connections: bool = False
    weight_norm: bool = False
    adaptive_activations: bool = False
    # bounds: List[Tuple[float, float]] = [(1.0, 1.0), (1.0, 1.0)]
    nr_levels: int = 5
    nr_features_per_level: int = 2
    log2_hashmap_size: int = 19
    base_resolution: int = 2
    finest_resolution: int = 32


@dataclass
class FNOConf(ModelConf):
    _target_: str = "modulus.models.fno.FNOArch"
    dimension: int = MISSING
    nr_fno_layers: int = 4
    fno_layer_size: int = 32
    fno_modes: Any = 16  # Change it Union[int, List[int]]
    padding: int = 8
    padding_type: str = "constant"
    output_fc_layer_sizes: List[int] = field(default_factory=lambda: [16])
    # activation_fn: Activation = Activation.GELU
    coord_features: bool = True


@dataclass
class AFNOConf(ModelConf):
    _target_: str = "modulus.models.afno.AFNOArch"
    img_shape: Tuple[int] = MISSING
    patch_size: int = 16
    embed_dim: int = 256
    depth: int = 4
    num_blocks: int = 8


@dataclass
class SRResConf(ModelConf):
    _target_: str = "modulus.models.super_res_net.SRResNetArch"
    large_kernel_size: int = 7
    small_kernel_size: int = 3
    conv_layer_size: int = 32
    n_resid_blocks: int = 8
    scaling_factor: int = 8
    # activation_fn: Activation = Activation.PRELU


@dataclass
class Pix2PixConf(ModelConf):
    _target_: str = "modulus.models.pix2pix.Pix2PixArch"
    dimension: int = MISSING
    conv_layer_size: int = 64
    n_downsampling: int = 3
    n_blocks: int = 3
    scaling_factor: int = 1
    batch_norm: bool = True
    padding_type: str = "reflect"
    # activation_fn: Activation = Activation.RELU


def register_arch_configs() -> None:
    # Information regarding multiple config groups
    # https://hydra.cc/docs/next/patterns/select_multiple_configs_from_config_group/
    cs = ConfigStore.instance()
    cs.store(
        group="arch",
        name="fully_connected",
        node={"fully_connected": FullyConnectedConf()},
    )

    cs.store(
        group="arch",
        name="fused_mlp",
        node={"fused_mlp": FusedMLPConf()},
    )

    cs.store(
        group="arch",
        name="fused_fourier_net",
        node={"fused_fourier_net": FusedFourierNetConf()},
    )

    cs.store(
        group="arch",
        name="fused_grid_encoding_net",
        node={"fused_grid_encoding_net": FusedGridEncodingNetConf()},
    )

    cs.store(
        group="arch",
        name="fourier_net",
        node={"fourier_net": FourierConf()},
    )

    cs.store(
        group="arch",
        name="highway_fourier",
        node={"highway_fourier": HighwayFourierConf()},
    )

    cs.store(
        group="arch",
        name="modified_fourier",
        node={"modified_fourier": ModifiedFourierConf()},
    )

    cs.store(
        group="arch",
        name="multiplicative_fourier",
        node={"multiplicative_fourier": MultiplicativeFilterConf()},
    )

    cs.store(
        group="arch",
        name="multiscale_fourier",
        node={"multiscale_fourier": MultiscaleFourierConf()},
    )

    cs.store(
        group="arch",
        name="siren",
        node={"siren": SirenConf()},
    )

    cs.store(
        group="arch",
        name="hash_net",
        node={"hash_net": MultiresolutionHashNetConf()},
    )

    cs.store(
        group="arch",
        name="fno",
        node={"fno": FNOConf()},
    )

    cs.store(
        group="arch",
        name="afno",
        node={"afno": AFNOConf()},
    )

    cs.store(
        group="arch",
        name="super_res",
        node={"super_res": SRResConf()},
    )

    cs.store(
        group="arch",
        name="pix2pix",
        node={"pix2pix": Pix2PixConf()},
    )

    # Schemas for extending models
    # Info: https://hydra.cc/docs/next/patterns/extending_configs/
    cs.store(
        group="arch_schema",
        name="fully_connected",
        node=FullyConnectedConf,
    )

    cs.store(
        group="arch_schema",
        name="fused_mlp",
        node=FusedMLPConf,
    )

    cs.store(
        group="arch_schema",
        name="fused_fourier_net",
        node=FusedFourierNetConf,
    )

    cs.store(
        group="arch_schema",
        name="fused_grid_encoding_net",
        node=FusedGridEncodingNetConf,
    )

    cs.store(
        group="arch_schema",
        name="fourier_net",
        node=FourierConf,
    )

    cs.store(
        group="arch_schema",
        name="highway_fourier",
        node=HighwayFourierConf,
    )

    cs.store(
        group="arch_schema",
        name="modified_fourier",
        node=ModifiedFourierConf,
    )

    cs.store(
        group="arch_schema",
        name="multiplicative_fourier",
        node=MultiplicativeFilterConf,
    )

    cs.store(
        group="arch_schema",
        name="multiscale_fourier",
        node=MultiscaleFourierConf,
    )

    cs.store(
        group="arch_schema",
        name="siren",
        node=SirenConf,
    )

    cs.store(
        group="arch_schema",
        name="hash_net",
        node=MultiresolutionHashNetConf,
    )

    cs.store(
        group="arch_schema",
        name="fno",
        node=FNOConf,
    )

    cs.store(
        group="arch_schema",
        name="afno",
        node=AFNOConf,
    )

    cs.store(
        group="arch_schema",
        name="super_res",
        node=SRResConf,
    )

    cs.store(
        group="arch_schema",
        name="pix2pix",
        node=Pix2PixConf,
    )
