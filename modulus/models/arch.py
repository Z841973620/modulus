import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import logging

from typing import List, Dict, Union, Tuple
from modulus.key import Key
from modulus.node import Node
from modulus.constants import JIT_PYTORCH_VERSION

logger = logging.getLogger(__name__)


class Arch(nn.Module):
    """
    Base class for all neural networks
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.periodicity = periodicity
        self.saveable = True

        self.input_key_dict = {str(var): var.size for var in input_keys}
        self.output_key_dict = {str(var): var.size for var in output_keys}

        self.input_scales = {str(k): k.scale for k in input_keys}
        self.output_scales = {str(k): k.scale for k in output_keys}

        self.detach_keys = detach_keys
        self.detach_key_dict: Dict[str, int] = {
            str(var): var.size for var in detach_keys
        }

        # If no detach keys, add a dummy for TorchScript compilation
        if not self.detach_key_dict:
            dummy_str = "_"
            while dummy_str in self.input_key_dict:
                dummy_str += "_"
            self.detach_key_dict[dummy_str] = 0

    def make_node(self, name: str, jit: bool = True, optimize: bool = True):
        """Makes neural network node for unrolling with Modulus `Graph`.

        Parameters
        ----------
        name : str
            This will be used as the name of created node.
        jit : bool
            If true the compile with jit, https://pytorch.org/docs/stable/jit.html.
        optimize : bool
            If true then treat parameters as optimizable.

        Examples
        --------
        Here is a simple example of creating a node from the fully connected network::

            >>> from .fully_connected import FullyConnectedArch
            >>> from modulus.key import Key
            >>> fc_arch = FullyConnectedArch([Key('x'), Key('y')], [Key('u')])
            >>> fc_node = fc_arch.make_node(name="fc_node")
            >>> print(fc_node)
            node: fc_node
            inputs: [x, y]
            derivatives: []
            outputs: [u]
            optimize: True

        """
        # set name for loading and saving model
        self.name = name
        self.checkpoint_filename = name + ".pth"

        # compile network
        if jit:
            # Warn user if pytorch version difference
            if not torch.__version__ == JIT_PYTORCH_VERSION:
                logger.warn(
                    f"Installed PyTorch version {torch.__version__} is not TorchScript"
                    + f" supported in Modulus. Version {JIT_PYTORCH_VERSION} is officially supported."
                )

            arch = torch.jit.script(self)
            node_name = "Arch Node (jit): " + "" if name is None else str(name)
            logger.info("Jit compiling network arch")
        else:
            arch = self
            node_name = "Arch Node: " + "" if name is None else str(name)

        # Set save and load methods TODO this is hacky but required for jit
        arch.save = self.save
        arch.load = self.load

        # Create and return node from this network architecture
        net_node = Node(
            self.input_keys, self.output_keys, arch, name=node_name, optimize=optimize
        )
        return net_node

    def save(self, directory):
        torch.save(self.state_dict(), directory + "/" + self.checkpoint_filename)

    def load(self, directory, map_location=None):
        self.load_state_dict(
            torch.load(
                directory + "/" + self.checkpoint_filename, map_location=map_location
            )
        )

    @staticmethod
    def prepare_input(
        input_variables: Dict[str, Tensor],
        mask: List[str],
        detach_dict: Dict[str, int],
        dim: int = 0,
        input_scales: Union[Dict[str, Tuple[float, float]], None] = None,
        periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
    ) -> Tensor:
        output_tensor = []
        for key in mask:
            if key in detach_dict:
                x = input_variables[key].detach()
            else:
                x = input_variables[key]
            # Scale input data
            if input_scales is not None:
                x = (x - input_scales[key][0]) / input_scales[key][1]

            append_tensor = [x]
            if periodicity is not None:
                if key in list(periodicity.keys()):
                    scaled_input = (x - periodicity[key][0]) / (
                        periodicity[key][1] - periodicity[key][0]
                    )
                    sin_tensor = torch.sin(2.0 * np.pi * scaled_input)
                    cos_tensor = torch.cos(2.0 * np.pi * scaled_input)
                    append_tensor = [sin_tensor, cos_tensor]
            output_tensor += append_tensor
        return torch.cat(output_tensor, dim=dim)

    @staticmethod
    def prepare_output(
        output_tensor: Tensor,
        output_var: Dict[str, int],
        dim: int = 0,
        output_scales: Union[Dict[str, Tuple[float, float]], None] = None,
    ) -> Dict[str, Tensor]:

        # create unnormalised output tensor
        output = {}
        for k, v in zip(
            output_var,
            torch.split(output_tensor, list(output_var.values()), dim=dim),
        ):
            output[k] = v
            if output_scales is not None:
                output[k] = output[k] * output_scales[k][1] + output_scales[k][0]

        return output
