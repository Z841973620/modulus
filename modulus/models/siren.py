from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

import modulus.models.layers.layers as layers
from .arch import Arch
from modulus.key import Key


class SirenArch(Arch):
    """Sinusoidal Representation Network (SIREN).

    Parameters
    ----------
    input_keys : List[Key]
        Input key list.
    output_keys : List[Key]
        Output key list.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    layer_size : int, optional
        Layer size for every hidden layer of the model, by default 512
    nr_layers : int, optional
        Number of hidden layers of the model, by default 6
    first_omega : float, optional
        Scales first weight matrix by this factor, by default 30
    omega : float, optional
        Scales the weight matrix of all hidden layers by this factor, by default 30
    normalization : Dict[str, Tuple[float, float]], optional
        Normalization of input to network, by default None


    Variable Shape
    --------------
    - Input variable tensor shape: :math:`[N, size]`
    - Output variable tensor shape: :math:`[N, size]`

    Example
    -------
    Siren model (2 -> 64 -> 64 -> 2)

    >>> arch = .siren.SirenArch(
    >>>    [Key("x", size=2)],
    >>>    [Key("y", size=2)],
    >>>    layer_size = 64,
    >>>    nr_layers = 2)
    >>> model = arch.make_node()
    >>> input = {"x": torch.randn(64, 2)}
    >>> output = model.evaluate(input)

    Note
    ----
    Reference: Sitzmann, Vincent, et al.
    Implicit Neural Representations with Periodic Activation Functions.
    https://arxiv.org/abs/2006.09661.
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_size: int = 512,
        nr_layers: int = 6,
        first_omega: float = 30.0,
        omega: float = 30.0,
        normalization: Dict[str, Tuple[float, float]] = None,
    ) -> None:
        super().__init__(
            input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys
        )

        in_features = sum(self.input_key_dict.values())
        out_features = sum(self.output_key_dict.values())

        layers_list = []

        layers_list.append(
            layers.SirenLayer(
                in_features,
                layer_size,
                layers.SirenLayerType.FIRST,
                first_omega,
            )
        )

        for _ in range(nr_layers - 1):
            layers_list.append(
                layers.SirenLayer(
                    layer_size, layer_size, layers.SirenLayerType.HIDDEN, omega
                )
            )

        layers_list.append(
            layers.SirenLayer(
                layer_size, out_features, layers.SirenLayerType.LAST, omega
            )
        )

        self.layers = nn.Sequential(*layers_list)
        self.normalization: Optional[Dict[str, Tuple[float, float]]] = normalization

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.prepare_input(
            self._normalize(in_vars, self.normalization),
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
        )
        x = self.layers(x)
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )

    def _normalize(
        self,
        in_vars: Dict[str, Tensor],
        norms: Optional[Dict[str, Tuple[float, float]]],
    ) -> Dict[str, Tensor]:
        if norms is None:
            return in_vars

        normalized_in_vars = {}
        for k, v in in_vars.items():
            if k in norms:
                v = (v - norms[k][0]) / (norms[k][1] - norms[k][0])
                v = 2 * v - 1
            normalized_in_vars[k] = v
        return normalized_in_vars
