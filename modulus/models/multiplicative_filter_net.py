import enum
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import modulus.models.layers.layers as layers
from .arch import Arch
from modulus.key import Key


class FilterType(enum.Enum):
    FOURIER = enum.auto()
    GABOR = enum.auto()


class MultiplicativeFilterNetArch(Arch):
    """
    Multiplicative Filter Net with Activations
    Reference: Fathony, R., Sahu, A.K., AI, A.A., Willmott, D. and Kolter, J.Z., MULTIPLICATIVE FILTER NETWORKS.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    layer_size : int = 512
        Layer size for every hidden layer of the model.
    nr_layers : int = 6
        Number of hidden layers of the model.
    skip_connections : bool = False
        If true then apply skip connections every 2 hidden layers.
    activation_fn : layers.Activation = layers.Activation.SILU
        Activation function used by network.
    filter_type : FilterType = FilterType.FOURIER
        Filter type for multiplicative filter network, (Fourier or Gabor).
    weight_norm : bool = True
        Use weight norm on fully connected layers.
    input_scale : float = 10.0
        Scale inputs for multiplicative filters.
    gabor_alpha : float = 6.0
        Alpha value for Gabor filter.
    gabor_beta : float = 1.0
        Beta value for Gabor filter.
    normalization : Optional[Dict[str, Tuple[float, float]]] = None
        Normalization of input to network.
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_size: int = 512,
        nr_layers: int = 6,
        skip_connections: bool = False,
        activation_fn=layers.Activation.IDENTITY,
        filter_type: FilterType = FilterType.FOURIER,
        weight_norm: bool = True,
        input_scale: float = 10.0,
        gabor_alpha: float = 6.0,
        gabor_beta: float = 1.0,
        normalization: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        super().__init__(
            input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys
        )

        in_features = sum(self.input_key_dict.values())
        out_features = sum(self.output_key_dict.values())

        self.nr_layers = nr_layers
        self.skip_connections = skip_connections

        if filter_type == FilterType.FOURIER:
            self.first_filter = layers.FourierFilter(
                in_features=in_features,
                layer_size=layer_size,
                nr_layers=nr_layers,
                input_scale=input_scale,
            )
        elif filter_type == FilterType.GABOR:
            self.first_filter = layers.GaborFilter(
                in_features=in_features,
                layer_size=layer_size,
                nr_layers=nr_layers,
                input_scale=input_scale,
                alpha=gabor_alpha,
                beta=gabor_beta,
            )
        else:
            raise ValueError

        self.filters = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        for i in range(nr_layers):
            self.fc_layers.append(
                layers.FCLayer(
                    in_features=layer_size,
                    out_features=layer_size,
                    activation_fn=activation_fn,
                    weight_norm=weight_norm,
                )
            )
            if filter_type == FilterType.FOURIER:
                self.filters.append(
                    layers.FourierFilter(
                        in_features=in_features,
                        layer_size=layer_size,
                        nr_layers=nr_layers,
                        input_scale=input_scale,
                    )
                )
            elif filter_type == FilterType.GABOR:
                self.filters.append(
                    layers.GaborFilter(
                        in_features=in_features,
                        layer_size=layer_size,
                        nr_layers=nr_layers,
                        input_scale=input_scale,
                        alpha=gabor_alpha,
                        beta=gabor_beta,
                    )
                )
            else:
                raise ValueError

        self.final_layer = layers.FCLayer(
            in_features=layer_size,
            out_features=out_features,
            activation_fn=layers.Activation.IDENTITY,
            weight_norm=False,
            activation_par=None,
        )

        self.normalization: Optional[Dict[str, Tuple[float, float]]] = normalization

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.prepare_input(
            self._normalize(in_vars, self.normalization),
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
        )

        res = self.first_filter(x)
        res_skip: Optional[Tensor] = None
        for i, (fc_layer, filter) in enumerate(zip(self.fc_layers, self.filters)):
            res_fc = fc_layer(res)
            res_filter = filter(x)
            res = res_fc * res_filter
            if self.skip_connections and i % 2 == 0:
                if res_skip is not None:
                    res, res_skip = res + res_skip, res
                else:
                    res_skip = res

        res = self.final_layer(res)
        return self.prepare_output(
            res, self.output_key_dict, dim=-1, output_scales=self.output_scales
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
