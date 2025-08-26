from typing import Dict
from typing import List

import torch
import torch.nn as nn
from torch import Tensor


import modulus.models.layers.layers as layers
from .arch import Arch
from modulus.key import Key


class RadialBasisArch(Arch):
    """
    Radial Basis Neural Network.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    bounds : Dict[str, Tuple[float, float]]
        Bounds to to randomly generate radial basis functions in.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    nr_centers : int = 128
        number of radial basis functions to use.
    sigma : float = 0.1
        Sigma in radial basis kernel.
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        bounds: Dict[str, List[float]],
        detach_keys: List[Key] = [],
        nr_centers: int = 128,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys
        )

        out_features = sum(self.output_key_dict.values())

        self.nr_centers = nr_centers
        self.sigma = sigma

        self.centers = nn.Parameter(
            torch.empty(nr_centers, len(bounds)), requires_grad=False
        )
        with torch.no_grad():
            for idx, bound in enumerate(bounds.values()):
                self.centers[:, idx].uniform_(bound[0], bound[1])

        self.fc_layer = layers.FCLayer(
            nr_centers,
            out_features,
            activation_fn=layers.Activation.IDENTITY,
        )

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.prepare_input(
            in_vars, self.input_key_dict.keys(), self.detach_key_dict, -1
        )
        shape = (x.size(0), self.nr_centers, x.size(1))
        x = x.unsqueeze(1).expand(shape)
        centers = self.centers.unsqueeze(0).expand(shape)

        radial_activation = torch.exp(
            -0.5 * torch.square(torch.norm(centers - x, p=2, dim=-1) / self.sigma)
        )

        x = self.fc_layer(radial_activation)
        return self.prepare_output(x, self.output_key_dict, -1)
