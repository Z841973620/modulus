from typing import Optional, Dict, Tuple, Union
from modulus.key import Key

import torch
import torch.nn as nn
from torch import Tensor

from .layers import layers
from .arch import Arch
from .fully_connected import FullyConnectedArch

from typing import List


class DeepONetArch(Arch):
    def __init__(
        self,
        dim: int,
        branch_net: Arch,
        trunk_net: Arch,
        branch_net_output_size: int,
        input_keys: List[Key] = None,
        output_keys: List[Key] = None,
        periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
        detach_keys: List[Key] = [],
    ) -> None:
        # verify the size of neural networks are allowed
        assert len(branch_net.output_keys) == len(
            trunk_net.output_keys
        ), "The outputs size of Branch net and Trunk net must be matched!"

        super().__init__(
            input_keys=branch_net.input_keys + trunk_net.input_keys,
            output_keys=output_keys,
            detach_keys=detach_keys,
            periodicity=periodicity,
        )

        # input parameter dimension
        self.dim = dim
        # branch net
        self.branch_net = branch_net
        self.branch_input_keys = self.branch_net.input_keys
        # trunk net
        self.trunk_net = trunk_net
        self.trunk_input_keys = self.trunk_net.input_keys

        if self.dim >= 2:
            self.linear_layer = nn.Linear(
                branch_net_output_size, sum(self.trunk_net.output_key_dict.values())
            )

    def make_node(self, name: str, jit: bool = False, optimize: bool = True):
        jit = False
        return super().make_node(name, jit, optimize)

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        branch_in_vars = {str(k): in_vars[str(k)] for k in self.branch_net.input_keys}
        trunk_in_vars = {str(k): in_vars[str(k)] for k in self.trunk_net.input_keys}

        branch_y = self.branch_net(branch_in_vars)
        trunk_y = self.trunk_net(trunk_in_vars)

        branch_y_tensor = branch_y[self.branch_net.output_keys[0].name]
        trunk_y_tensor = trunk_y[self.trunk_net.output_keys[0].name]

        if self.dim >= 2:
            # flatten high-dimensional tensors into 2D tensors
            branch_y_tensor = branch_y_tensor.reshape(branch_y_tensor.shape[0], -1)
            branch_y_tensor = self.linear_layer(branch_y_tensor)

        y = torch.sum(branch_y_tensor * trunk_y_tensor, dim=-1, keepdim=True)

        return self.prepare_output(
            y, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )
