from typing import Dict, List, Union, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import logging

import modulus.models.layers.layers as layers
from modulus.models.layers.layers import Activation
from .arch import Arch
from modulus.key import Key
from modulus.node import Node
from modulus.constants import JIT_PYTORCH_VERSION

logger = logging.getLogger(__name__)


class FNO1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        nr_fno_layers: int = 4,
        fno_layer_size: int = 32,
        fno_modes: Union[int, List[int]] = 16,
        padding: int = 8,
        padding_type: str = "constant",
        output_fc_layer_sizes: List[int] = [32],
        activation_fn: Activation = Activation.GELU,
        coord_features: bool = True,
        squeeze_latent_size: Union[int, None] = None,
    ) -> None:
        super().__init__()

        self.encoder = self.Encoder(
            in_channels=in_channels,
            nr_fno_layers=nr_fno_layers,
            fno_layer_size=fno_layer_size,
            fno_modes=fno_modes,
            padding=padding,
            padding_type=padding_type,
            activation_fn=activation_fn,
            coord_features=coord_features,
            squeeze_latent_size=squeeze_latent_size,
        )

        self.decoder = self.Decoder(
            out_channels=out_channels,
            fno_layer_size=fno_layer_size,
            output_fc_layer_sizes=output_fc_layer_sizes,
            activation_fn=activation_fn,
            squeeze_latent_size=squeeze_latent_size,
        )

    class Encoder(nn.Module):
        # 1D spectral encoder

        def __init__(
            self,
            in_channels: int = 1,
            nr_fno_layers: int = 4,
            fno_layer_size: int = 32,
            fno_modes: Union[int, List[int]] = 16,
            padding: int = 8,
            padding_type: str = "constant",
            activation_fn: Activation = Activation.GELU,
            coord_features: bool = True,
            squeeze_latent_size: Union[int, None] = None,
        ) -> None:
            super().__init__()

            self.in_channels = in_channels
            self.nr_fno_layers = nr_fno_layers
            self.fno_width = fno_layer_size
            self.coord_features = coord_features
            # Spectral modes to have weights
            if isinstance(fno_modes, int):
                fno_modes = [fno_modes]
            # Add relative coordinate feature
            if self.coord_features:
                self.in_channels = self.in_channels + 1
            self.activation_fn = layers.get_activation_fn(activation_fn)

            self.spconv_layers = nn.ModuleList()
            self.conv_layers = nn.ModuleList()

            # Initial lift layer
            self.lift_layer = layers.Conv1dFCLayer(self.in_channels, self.fno_width)

            # Build Neural Fourier Operators
            for _ in range(self.nr_fno_layers):
                self.spconv_layers.append(
                    layers.SpectralConv1d(self.fno_width, self.fno_width, fno_modes[0])
                )
                self.conv_layers.append(nn.Conv1d(self.fno_width, self.fno_width, 1))

            if squeeze_latent_size is not None:
                self.squeeze_layer = layers.Conv1dFCLayer(
                    self.fno_width, squeeze_latent_size
                )
            else:
                self.squeeze_layer = None

            # Padding values for spectral conv
            if isinstance(padding, int):
                padding = [padding]
            self.pad = padding[:1]
            self.ipad = [-pad if pad > 0 else None for pad in self.pad]
            self.padding_type = padding_type

        def forward(self, x: Tensor) -> Tensor:

            if self.coord_features:
                coord_feat = self.meshgrid(list(x.shape), x.device)
                x = torch.cat((x, coord_feat), dim=1)

            x = self.lift_layer(x)
            # (left, right)
            x = F.pad(x, (0, self.pad[0]), mode=self.padding_type)
            # Spectral layers
            for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
                conv, w = conv_w
                if k < len(self.conv_layers) - 1:
                    x = self.activation_fn(
                        conv(x) + w(x)
                    )  # Spectral Conv + GELU causes JIT issue!
                else:
                    x = conv(x) + w(x)

            if self.squeeze_layer is not None:
                x = self.squeeze_layer(x)

            x = x[..., : self.ipad[0]]
            return x

        def meshgrid(self, shape: List[int], device: torch.device):
            bsize, size_x = shape[0], shape[2]
            grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1)
            return grid_x

    class Decoder(nn.Module):
        # Fully connected decoder

        def __init__(
            self,
            out_channels: int = 1,
            fno_layer_size: int = 32,
            output_fc_layer_sizes: List[int] = [16],
            activation_fn: Activation = Activation.GELU,
            squeeze_latent_size: Union[int, None] = None,
        ) -> None:
            super().__init__()
            self.out_channels = out_channels
            self.fno_width = fno_layer_size
            self.activation_fn = layers.get_activation_fn(activation_fn)

            self.fc_layers = nn.ModuleList()
            # Build output fully-connected layers
            output_fc_layer_sizes = [self.fno_width] + output_fc_layer_sizes
            if squeeze_latent_size is not None:
                output_fc_layer_sizes = [squeeze_latent_size] + output_fc_layer_sizes
            for i in range(1, len(output_fc_layer_sizes)):
                self.fc_layers.append(
                    layers.Conv1dFCLayer(
                        in_channels=output_fc_layer_sizes[i - 1],
                        out_channels=output_fc_layer_sizes[i],
                        activation_fn=activation_fn,
                    )
                )
            self.fc_layers.append(
                layers.Conv1dFCLayer(
                    in_channels=output_fc_layer_sizes[-1],
                    out_channels=self.out_channels,
                )
            )

        def forward(self, x: Tensor) -> Tensor:
            # Output fully-connected layers
            for fc in self.fc_layers:
                x = fc(x)
            return x

    def forward(self, x: Tensor) -> Tensor:

        assert (
            x.dim() == 3
        ), "Only 3D tensors [batch, in_channels, grid] accepted for 1D FNO"
        # Spectral encoder
        x = self.encoder(x)
        # Fully-connected point-wise decoder
        x = self.decoder(x)

        return x


class FNO2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        nr_fno_layers: int = 4,
        fno_layer_size: int = 32,
        fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        output_fc_layer_sizes: List[int] = [16],
        activation_fn: Activation = Activation.GELU,
        coord_features: bool = True,
        squeeze_latent_size: Union[int, None] = None,
    ) -> None:
        super().__init__()

        self.encoder = self.Encoder(
            in_channels=in_channels,
            nr_fno_layers=nr_fno_layers,
            fno_layer_size=fno_layer_size,
            fno_modes=fno_modes,
            padding=padding,
            padding_type=padding_type,
            activation_fn=activation_fn,
            coord_features=coord_features,
            squeeze_latent_size=squeeze_latent_size,
        )

        self.decoder = self.Decoder(
            out_channels=out_channels,
            fno_layer_size=fno_layer_size,
            output_fc_layer_sizes=output_fc_layer_sizes,
            activation_fn=activation_fn,
            squeeze_latent_size=squeeze_latent_size,
        )

    class Encoder(nn.Module):
        # 2D spectral encoder

        def __init__(
            self,
            in_channels: int = 1,
            nr_fno_layers: int = 4,
            fno_layer_size: int = 32,
            fno_modes: Union[int, List[int]] = 16,
            padding: Union[int, List[int]] = 8,
            padding_type: str = "constant",
            activation_fn: Activation = Activation.GELU,
            coord_features: bool = True,
            squeeze_latent_size: Union[int, None] = None,
        ) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.nr_fno_layers = nr_fno_layers
            self.fno_width = fno_layer_size
            self.coord_features = coord_features
            # Spectral modes to have weights
            if isinstance(fno_modes, int):
                fno_modes = [fno_modes, fno_modes]
            # Add relative coordinate feature
            if self.coord_features:
                self.in_channels = self.in_channels + 2
            self.activation_fn = layers.get_activation_fn(activation_fn)

            self.spconv_layers = nn.ModuleList()
            self.conv_layers = nn.ModuleList()

            # Initial lift layer
            self.lift_layer = layers.Conv2dFCLayer(self.in_channels, self.fno_width)

            # Build Neural Fourier Operators
            for _ in range(self.nr_fno_layers):
                self.spconv_layers.append(
                    layers.SpectralConv2d(
                        self.fno_width, self.fno_width, fno_modes[0], fno_modes[1]
                    )
                )
                self.conv_layers.append(nn.Conv2d(self.fno_width, self.fno_width, 1))

            if squeeze_latent_size is not None:
                self.squeeze_layer = layers.Conv2dFCLayer(
                    self.fno_width, squeeze_latent_size
                )
            else:
                self.squeeze_layer = None

            # Padding values for spectral conv
            if isinstance(padding, int):
                padding = [padding, padding]
            padding = padding + [0, 0]  # Pad with zeros for smaller lists
            self.pad = padding[:2]
            self.ipad = [-pad if pad > 0 else None for pad in self.pad]
            self.padding_type = padding_type

        def forward(self, x: Tensor) -> Tensor:

            if self.coord_features:
                coord_feat = self.meshgrid(list(x.shape), x.device)
                x = torch.cat((x, coord_feat), dim=1)

            x = self.lift_layer(x)
            # (left, right, top, bottom)
            x = F.pad(x, (0, self.pad[0], 0, self.pad[1]), mode=self.padding_type)
            # Spectral layers
            for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
                conv, w = conv_w
                if k < len(self.conv_layers) - 1:
                    x = self.activation_fn(
                        conv(x) + w(x)
                    )  # Spectral Conv + GELU causes JIT issue!
                else:
                    x = conv(x) + w(x)

            if self.squeeze_layer is not None:
                x = self.squeeze_layer(x)

            # remove padding
            x = x[..., : self.ipad[1], : self.ipad[0]]

            return x

        def meshgrid(self, shape: List[int], device: torch.device):
            bsize, size_x, size_y = shape[0], shape[2], shape[3]
            grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
            grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
            grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
            grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
            return torch.cat((grid_x, grid_y), dim=1)

    class Decoder(nn.Module):
        # Fully connected decoder

        def __init__(
            self,
            out_channels: int = 1,
            fno_layer_size: int = 32,
            output_fc_layer_sizes: List[int] = [16],
            activation_fn: Activation = Activation.GELU,
            squeeze_latent_size: Union[int, None] = None,
        ) -> None:
            super().__init__()
            self.out_channels = out_channels
            self.fno_width = fno_layer_size
            self.activation_fn = layers.get_activation_fn(activation_fn)

            self.fc_layers = nn.ModuleList()
            # Build output fully-connected layers
            output_fc_layer_sizes = [self.fno_width] + output_fc_layer_sizes
            if squeeze_latent_size is not None:
                output_fc_layer_sizes = [squeeze_latent_size] + output_fc_layer_sizes
            for i in range(1, len(output_fc_layer_sizes)):
                self.fc_layers.append(
                    layers.Conv2dFCLayer(
                        in_channels=output_fc_layer_sizes[i - 1],
                        out_channels=output_fc_layer_sizes[i],
                        activation_fn=activation_fn,
                    )
                )
            self.fc_layers.append(
                layers.Conv2dFCLayer(
                    in_channels=output_fc_layer_sizes[-1],
                    out_channels=self.out_channels,
                )
            )

        def forward(self, x: Tensor) -> Tensor:
            # Output fully-connected layers
            for fc in self.fc_layers:
                x = fc(x)
            return x

    def forward(self, x: Tensor) -> Tensor:

        assert (
            x.dim() == 4
        ), "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D FNO"
        # Spectral encoder
        x = self.encoder(x)
        # Fully-connected point-wise decoder
        x = self.decoder(x)

        return x


class FNO3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        nr_fno_layers: int = 4,
        fno_layer_size: int = 32,
        fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        output_fc_layer_sizes: List[int] = [16],
        activation_fn: Activation = Activation.GELU,
        coord_features: bool = True,
        squeeze_latent_size: Union[int, None] = None,
    ) -> None:
        super().__init__()

        self.encoder = self.Encoder(
            in_channels=in_channels,
            nr_fno_layers=nr_fno_layers,
            fno_layer_size=fno_layer_size,
            fno_modes=fno_modes,
            padding=padding,
            padding_type=padding_type,
            activation_fn=activation_fn,
            coord_features=coord_features,
            squeeze_latent_size=squeeze_latent_size,
        )

        self.decoder = self.Decoder(
            out_channels=out_channels,
            fno_layer_size=fno_layer_size,
            output_fc_layer_sizes=output_fc_layer_sizes,
            activation_fn=activation_fn,
            squeeze_latent_size=squeeze_latent_size,
        )

    class Encoder(nn.Module):
        # 3D spectral encoder

        def __init__(
            self,
            in_channels: int = 1,
            nr_fno_layers: int = 4,
            fno_layer_size: int = 32,
            fno_modes: Union[int, List[int]] = 16,
            padding: Union[int, List[int]] = 8,
            padding_type: str = "constant",
            activation_fn: Activation = Activation.GELU,
            coord_features: bool = True,
            squeeze_latent_size: Union[int, None] = None,
        ) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.nr_fno_layers = nr_fno_layers
            self.fno_width = fno_layer_size
            self.coord_features = coord_features
            # Spectral modes to have weights
            if isinstance(fno_modes, int):
                fno_modes = [fno_modes, fno_modes, fno_modes]
            # Add relative coordinate feature
            if self.coord_features:
                self.in_channels = self.in_channels + 3
            self.activation_fn = layers.get_activation_fn(activation_fn)

            self.spconv_layers = nn.ModuleList()
            self.conv_layers = nn.ModuleList()

            # Initial lift layer
            self.lift_layer = layers.Conv3dFCLayer(self.in_channels, self.fno_width)

            # Build Neural Fourier Operators
            for _ in range(self.nr_fno_layers):
                self.spconv_layers.append(
                    layers.SpectralConv3d(
                        self.fno_width,
                        self.fno_width,
                        fno_modes[0],
                        fno_modes[1],
                        fno_modes[2],
                    )
                )
                self.conv_layers.append(nn.Conv3d(self.fno_width, self.fno_width, 1))

            if squeeze_latent_size is not None:
                self.squeeze_layer = layers.Conv3dFCLayer(
                    self.fno_width, squeeze_latent_size
                )
            else:
                self.squeeze_layer = None

            # Padding values for spectral conv
            if isinstance(padding, int):
                padding = [padding, padding, padding]
            padding = padding + [0, 0, 0]  # Pad with zeros for smaller lists
            self.pad = padding[:3]
            self.ipad = [-pad if pad > 0 else None for pad in self.pad]
            self.padding_type = padding_type

        def forward(self, x: Tensor) -> Tensor:

            if self.coord_features:
                coord_feat = self.meshgrid(list(x.shape), x.device)
                x = torch.cat((x, coord_feat), dim=1)

            x = self.lift_layer(x)
            # (left, right, top, bottom, front, back)
            x = F.pad(
                x,
                (0, self.pad[0], 0, self.pad[1], 0, self.pad[2]),
                mode=self.padding_type,
            )
            # Spectral layers
            for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
                conv, w = conv_w
                if k < len(self.conv_layers) - 1:
                    x = self.activation_fn(
                        conv(x) + w(x)
                    )  # Spectral Conv + GELU causes JIT issue!
                else:
                    x = conv(x) + w(x)

            if self.squeeze_layer is not None:
                x = self.squeeze_layer(x)

            x = x[..., : self.ipad[2], : self.ipad[1], : self.ipad[0]]
            return x

        def meshgrid(self, shape: List[int], device: torch.device):
            bsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]
            grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
            grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
            grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
            grid_x, grid_y, grid_z = torch.meshgrid(
                grid_x, grid_y, grid_z, indexing="ij"
            )
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
            grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
            grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
            return torch.cat((grid_x, grid_y, grid_z), dim=1)

    class Decoder(nn.Module):
        # Fully connected decoder

        def __init__(
            self,
            out_channels: int = 1,
            fno_layer_size: int = 32,
            output_fc_layer_sizes: List[int] = [16],
            activation_fn: Activation = Activation.GELU,
            squeeze_latent_size: Union[int, None] = None,
        ) -> None:
            super().__init__()
            self.out_channels = out_channels
            self.fno_width = fno_layer_size
            self.activation_fn = layers.get_activation_fn(activation_fn)

            self.fc_layers = nn.ModuleList()
            # Build output fully-connected layers
            output_fc_layer_sizes = [self.fno_width] + output_fc_layer_sizes
            if squeeze_latent_size is not None:
                output_fc_layer_sizes = [squeeze_latent_size] + output_fc_layer_sizes
            for i in range(1, len(output_fc_layer_sizes)):
                self.fc_layers.append(
                    layers.Conv3dFCLayer(
                        in_channels=output_fc_layer_sizes[i - 1],
                        out_channels=output_fc_layer_sizes[i],
                        activation_fn=activation_fn,
                    )
                )
            self.fc_layers.append(
                layers.Conv3dFCLayer(
                    in_channels=output_fc_layer_sizes[-1],
                    out_channels=self.out_channels,
                )
            )

        def forward(self, x: Tensor) -> Tensor:
            # Output fully-connected layers
            for fc in self.fc_layers:
                x = fc(x)

            return x

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.dim() == 5
        ), "Only 5D tensors [batch, in_channels, grid_x, grid_y, grid_z] accepted for 3D FNO"
        # Spectral encoder
        x = self.encoder(x)
        # Fully-connected point-wise decoder
        x = self.decoder(x)

        return x


class FNOArch(Arch):
    """Fourier neural operator (FNO) model.

    Note
    ----
    The FNO architecture supports options for 1D, 2D and 3D fields which can
    be controlled using the `dimension` parameter.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list. The key dimension size should equal the variables channel dim.
    output_keys : List[Key]
        Output key list. The key dimension size should equal the variables channel dim.
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    nr_fno_layers : int, optional
        Number of spectral convolution layers, by default 4
    fno_layer_size : int, optional
        Size of latent variables inside spectral convolutions, by default 32
    fno_modes : Union[int, List[int]], optional
        Number of Fourier modes with learnable weights, by default 16
    padding : int, optional
        Padding size for FFT calculations, by default 8
    padding_type : str, optional
        Padding type for FFT calculations ('constant', 'reflect', 'replicate'
        or 'circular'), by default "constant"
    output_fc_layer_sizes : List[int], optional
        List of point-wise fully connected decoder layers, by default [16]
    activation_fn : Activation, optional
        Activation function, by default Activation.GELU
    coord_features : bool, optional
        Use coordinate meshgrid as additional input feature, by default True
    domain_length : List[float], optional
        List defining the rectangular domain size, by default [1.0, 1.0]

    Variable Shape
    --------------
    Input variable tensor shape:

    - 1D: :math:`[N, size, W]`
    - 2D: :math:`[N, size, H, W]`
    - 3D: :math:`[N, size, D, H, W]`

    Output variable tensor shape:

    - 1D: :math:`[N, size,  W]`
    - 2D: :math:`[N, size, H, W]`
    - 3D: :math:`[N, size, D, H, W]`

    Example
    -------
    1D FNO model

    >>> fno_1d = .fno.FNOArch([Key("x", size=2)], [Key("y", size=2)], dimension=1)
    >>> model = fno_1d.make_node()
    >>> input = {"x": torch.randn(20, 2, 64)}
    >>> output = model.evaluate(input)

    2D FNO model

    >>> fno_2d = .fno.FNOArch([Key("x", size=2)], [Key("y", size=2)], dimension=2)
    >>> model = fno_2d.make_node()
    >>> input = {"x": torch.randn(20, 2, 64, 64)}
    >>> output = model.evaluate(input)

    3D FNO model

    >>> fno_3d = .fno.FNOArch([Key("x", size=2)], [Key("y", size=2)], dimension=3)
    >>> model = fno_3d.make_node()
    >>> input = {"x": torch.randn(20, 2, 64, 64, 64)}
    >>> output = model.evaluate(input)
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        dimension: int,
        detach_keys: List[Key] = [],
        nr_fno_layers: int = 4,
        fno_layer_size: int = 32,
        fno_modes: Union[int, List[int]] = 16,
        padding: int = 8,
        padding_type: str = "constant",
        output_fc_layer_sizes: List[int] = [16],
        activation_fn: Activation = Activation.GELU,
        coord_features: bool = True,
        domain_length: List[float] = [1.0, 1.0],
        squeeze_latent_size: Union[int, None] = None,
    ) -> None:
        super().__init__(input_keys=input_keys, output_keys=output_keys)
        self.derivative_keys = []
        self.domain_length = domain_length
        for var in self.output_keys:
            dx_name = str(var).split("__")  # Split name to get original var names
            if len(dx_name) == 2:  # First order
                assert (
                    dx_name[1] in ["x", "y", "z"][:dimension]
                ), f"Invalid first-order derivative {str(var)} for {dimension}d FNO"
                self.derivative_keys.append(var)
            elif len(dx_name) == 3:
                assert (
                    dx_name[1] in ["x", "y", "z"][:dimension]
                    and dx_name[1] == dx_name[2]
                ), f"Invalid second-order derivative {str(var)} for {dimension}d FNO"
                self.derivative_keys.append(var)
            elif len(dx_name) > 3:
                raise ValueError(
                    "FNO only supports first order and laplacian second order derivatives"
                )
        self.detach_keys = detach_keys
        if squeeze_latent_size is not None:
            self.latent_keys = [Key("fno_zeta", size=squeeze_latent_size)]
        else:
            self.latent_keys = [Key("fno_zeta", size=fno_layer_size)]
        self.derivative_key_dict = {str(var): var.size for var in self.derivative_keys}

        in_channels = sum(self.input_key_dict.values())
        out_channels = sum(self.output_key_dict.values()) - sum(
            self.derivative_key_dict.values()
        )

        if dimension == 1:
            FNOModel = FNO1D
        elif dimension == 2:
            FNOModel = FNO2D
        elif dimension == 3:
            FNOModel = FNO3D
        else:
            raise NotImplementedError(
                "Invalid dimensionality. Only 1D, 2D and 3D FNO implemented"
            )

        self._impl = FNOModel(
            in_channels,
            out_channels,
            nr_fno_layers=nr_fno_layers,
            fno_layer_size=fno_layer_size,
            fno_modes=fno_modes,
            padding=padding,
            padding_type=padding_type,
            output_fc_layer_sizes=output_fc_layer_sizes,
            activation_fn=activation_fn,
            coord_features=coord_features,
            squeeze_latent_size=squeeze_latent_size,
        )

    class EncoderArch(Arch):
        # Encoder Arch for constructing seperate node

        def __init__(
            self,
            input_keys: List[Key],
            output_keys: List[Key],
            encoder_model: nn.Module,
        ):
            super().__init__(input_keys=input_keys, output_keys=output_keys)
            self._impl = encoder_model

        def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:

            x = self.prepare_input(
                in_vars,
                self.input_key_dict.keys(),
                detach_dict=self.detach_key_dict,
                dim=1,
                input_scales=self.input_scales,
            )
            y = self._impl(x)

            return self.prepare_output(y, self.output_key_dict, dim=1)

    class DecoderArch(Arch):
        # Decoder Arch for constructing seperate node

        derivative_key_dict: Dict[str, int]  # Torch script need explicit type

        def __init__(
            self,
            input_keys: List[Key],
            output_keys: List[Key],
            derivative_keys: List[Key],
            domain_length: List[int],
            dcoder_model: nn.Module,
        ):
            super().__init__(input_keys=input_keys, output_keys=output_keys)
            self._impl = dcoder_model
            self.domain_length = domain_length
            self.derivative_keys = derivative_keys
            self.derivative_key_dict = {
                str(var): var.size for var in self.derivative_keys
            }
            # Pop derivative keys from FNO outputs
            for key in self.derivative_key_dict.keys():
                self.output_key_dict.pop(key, None)

            # Explicit gradient flags
            self.calc_dx = False
            self.calc_ddx = False
            for key in self.derivative_key_dict.keys():
                if len(key.split("__")) == 2:  # First order
                    self.calc_dx = True
                elif len(key.split("__")) == 3:  # Second order
                    self.calc_ddx = True

        def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
            # Split latent variables into seperate channels, then combine
            # This is needed to calculate the Hessian of the latent variables
            # with auto-grad
            x_list = torch.split(in_vars["fno_zeta"], 1, dim=1)
            if self.calc_dx or self.calc_ddx:
                in_vars["fno_zeta"] = torch.cat(x_list, dim=1)  # JIT issue here

            x = self.prepare_input(
                in_vars,
                self.input_key_dict.keys(),
                detach_dict=self.detach_key_dict,
                dim=1,
            )
            dim = int(len(x.shape) - 2)

            # Forward pass of the decoder NN
            y = self._impl(x)
            y = self.prepare_output(
                y, self.output_key_dict, dim=1, output_scales=self.output_scales
            )

            # PINO exact derivatives
            if self.calc_dx or self.calc_ddx:

                dx_list, ddx_list = self.calc_latent_derivatives(x, dim)

                y_dx = self.calc_derivatives(x, y, x_list, dx_list, ddx_list, dim)
                y.update(y_dx)

            return y

        @torch.jit.ignore
        def calc_latent_derivatives(
            self, x: Tensor, dim: int = 2
        ) -> Tuple[List[Tensor], List[Tensor]]:
            # Compute derivatives of latent variables via fourier methods
            # Padd domain by factor of 2 for non-periodic domains
            padd = [(i - 1) // 2 for i in list(x.shape[2:])]
            # Scale domain length by padding amount
            domain_length = [
                self.domain_length[i] * (2 * padd[i] + x.shape[i + 2]) / x.shape[i + 2]
                for i in range(dim)
            ]
            padding = padd + padd
            x_p = F.pad(x, padding, mode="replicate")
            dx, ddx = layers.fourier_derivatives(x_p, domain_length)
            # Trim padded domain
            if len(x.shape) == 3:
                dx = dx[..., padd[0] : -padd[0]]
                ddx = ddx[..., padd[0] : -padd[0]]
                dx_list = torch.split(dx, x.shape[1], dim=1)
                ddx_list = torch.split(ddx, x.shape[1], dim=1)
            elif len(x.shape) == 4:
                dx = dx[..., padd[0] : -padd[0], padd[1] : -padd[1]]
                ddx = ddx[..., padd[0] : -padd[0], padd[1] : -padd[1]]
                dx_list = torch.split(dx, x.shape[1], dim=1)
                ddx_list = torch.split(ddx, x.shape[1], dim=1)
            else:
                dx = dx[..., padd[0] : -padd[0], padd[1] : -padd[1], padd[2] : -padd[2]]
                ddx = ddx[
                    ..., padd[0] : -padd[0], padd[1] : -padd[1], padd[2] : -padd[2]
                ]
                dx_list = torch.split(dx, x.shape[1], dim=1)
                ddx_list = torch.split(ddx, x.shape[1], dim=1)

            return dx_list, ddx_list

        @torch.jit.ignore
        def calc_derivatives(
            self,
            x: Tensor,
            y: Dict[str, Tensor],
            x_list: List[Tensor],
            dx_list: List[Tensor],
            ddx_list: List[Tensor],
            dim: int = 2,
        ) -> Dict[str, Tensor]:

            # Loop through output variables independently
            y_out: Dict[str, Tensor] = {}
            for key in self.output_key_dict.keys():
                # First-order grads with back-prop
                outputs: List[torch.Tensor] = [y[key]]
                inputs: List[torch.Tensor] = [x]
                grad_outputs: List[Optional[torch.Tensor]] = [
                    torch.ones_like(y[key], device=y[key].device)
                ]
                dydzeta = torch.autograd.grad(
                    outputs,
                    inputs,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                for i, axis in enumerate(["x", "y", "z"]):
                    if f"{key}__{axis}" in self.derivative_key_dict:
                        # Chain rule: g'(x)*f'(g(x))
                        y_out[f"{key}__{axis}"] = torch.sum(
                            dx_list[i] * dydzeta, dim=1, keepdim=True
                        )

                # Calc second order if needed
                if self.calc_ddx:
                    y_ddx = self.calc_second_order_derivatives(
                        x, key, x_list, dx_list, ddx_list, dydzeta, dim
                    )
                    y_out.update(y_ddx)

            return y_out

        @torch.jit.ignore
        def calc_second_order_derivatives(
            self,
            x: Tensor,
            key: str,
            x_list: List[Tensor],
            dx_list: List[Tensor],
            ddx_list: List[Tensor],
            dydzeta: Tensor,
            dim: int = 2,
        ) -> Dict[str, Tensor]:

            # Brute force Hessian calc with auto-diff
            hessian = torch.zeros(
                dydzeta.shape[0],
                dydzeta.shape[1],
                dydzeta.shape[1],
                dydzeta.shape[2],
                dydzeta.shape[3],
            ).to(x.device)
            grad_outputs: List[Optional[torch.Tensor]] = [
                torch.ones_like(dydzeta[:, :1], device=dydzeta.device)
            ]
            for i in range(dydzeta.shape[1]):
                for j in range(i, dydzeta.shape[1]):
                    dyydzeta = torch.autograd.grad(
                        dydzeta[:, j : j + 1],
                        x_list[i],
                        grad_outputs=grad_outputs,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    if dyydzeta is not None:
                        hessian[:, i, j] = dyydzeta.squeeze(1)
                        hessian[:, j, i] = dyydzeta.squeeze(1)

            # Loop through output variables independently
            y_out: Dict[str, Tensor] = {}
            # Add needed derivatives
            for i, axis in enumerate(["x", "y", "z"]):
                if f"{key}__{axis}__{axis}" in self.derivative_key_dict:
                    dim_str = "ijk"[:dim]
                    # Chain rule: g''(x)*f'(g(x)) + g'(x)*f''(g(x))*g'(x)
                    y_out[f"{key}__{axis}__{axis}"] = (
                        torch.sum(ddx_list[i] * dydzeta, dim=1, keepdim=True)
                        + torch.einsum(
                            f"bm{dim_str},bmn{dim_str},bn{dim_str}->b{dim_str}",
                            dx_list[i],
                            hessian,
                            dx_list[i],
                        ).unsqueeze(1)
                    )

            return y_out

    def make_nodes(
        self, name: str, jit: bool = False, optimize: bool = True
    ) -> List[Node]:
        """Access Fourier Neural Operator as two nodes.

        Provides Fourier Neural Operator in two seperate spectral encoder and decoder
        nodes. This should be used for PINO, when "exact" gradient methods wants to be
        which require gradients to be calculated inbetween the encoder and decoder.

        Parameters
        ----------
        name : str
            This will be used as the name of created node.
        jit : bool
            If true the compile with jit, https://pytorch.org/docs/stable/jit.html.
        optimize : bool
            If true then treat parameters as optimizable.

        Returns
        -------
        List[Node]
            [Encoder nodes, Decoder nodes]
        """
        # Forcing JIT off for PINO
        jit = False

        # We want two nodes to create two sub archs (encoder and decoder)
        self.encoder = self.EncoderArch(
            self.input_keys, self.latent_keys, self._impl.encoder
        )

        self.decoder = self.DecoderArch(
            self.latent_keys,
            self.output_keys,
            self.derivative_keys,
            self.domain_length,
            self._impl.decoder,
        )

        self.encoder.name = name + "_enc"
        self.decoder.name = name + "_dec"

        self.encoder.checkpoint_filename = self.encoder.name + ".pth"
        self.decoder.checkpoint_filename = self.decoder.name + ".pth"

        # compile network
        if jit:
            # Warn user if pytorch version difference
            if not torch.__version__ == JIT_PYTORCH_VERSION:
                logger.warn(
                    f"Installed PyTorch version {torch.__version__} is not TorchScript"
                    + f" supported in Modulus. Version {JIT_PYTORCH_VERSION} is officially supported."
                )

            enc_arch = torch.jit.script(self.encoder)
            dec_arch = torch.jit.script(self.decoder)

            # set name for loading and saving model
            enc_node_name = (
                "Arch Node (jit): " + "enc" if name is None else str(enc_arch.name)
            )
            dec_node_name = (
                "Arch Node: " + "dec" if name is None else str(enc_arch.name)
            )
            logger.info("Jit compiling network arch")
        else:
            enc_arch = self.encoder
            dec_arch = self.decoder

            # set name for loading and saving model
            enc_arch.name = name + "_enc"
            dec_arch.name = name + "_dec"

            enc_node_name = (
                "Arch Node: " + "enc" if name is None else str(enc_arch.name)
            )
            dec_node_name = (
                "Arch Node: " + "dec" if name is None else str(enc_arch.name)
            )

        # Set save and load methods TODO this is hacky but required for jit
        enc_arch.save = self.encoder.save
        enc_arch.load = self.encoder.load

        dec_arch.save = self.decoder.save
        dec_arch.load = self.decoder.load

        # Create and return node from this network architecture
        enc_net_node = Node(
            self.input_keys,
            self.latent_keys,
            enc_arch,
            name=enc_node_name,
            optimize=optimize,
        )
        dec_net_node = Node(
            self.latent_keys,
            self.output_keys,
            dec_arch,
            name=dec_node_name,
            optimize=optimize,
        )
        return [enc_net_node, dec_net_node]

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_keys,
            dim=1,
            input_scales=self.input_scales,
        )
        y = self._impl(x)
        return self.prepare_output(
            y, self.output_key_dict, dim=1, output_scales=self.output_scales
        )
