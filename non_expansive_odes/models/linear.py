from abc import ABC, abstractmethod
from typing import Optional, Tuple
from warnings import warn

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv1d, Conv2d, Conv3d, Linear

from non_expansive_odes.utils.power_method import power_method_nonsquare


class ModuleTranspose(ABC):
    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def tangent(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def transpose(self, z: Tensor) -> Tensor:
        pass

    @abstractmethod
    def _generate_random_input(self) -> Tensor:
        pass

    def rescale(self, alpha: float = 1.0) -> None:
        with torch.no_grad():
            self.weight *= alpha / self.norm
        self._norm = alpha

    def compute_spectral_norm(
        self, u_init: Optional[Tensor] = None, k: int = 1
    ) -> float:
        if u_init is None and hasattr(self, "u"):
            u_init = self.u
        elif u_init is None:
            u_init = self._generate_random_input()
        with torch.no_grad():
            norm, self.u = power_method_nonsquare(
                self.tangent, self.transpose, u_init, k
            )
        self._norm = norm
        return norm

    @property
    def norm(self):
        return self._norm


class LinearNormalised(Linear, ModuleTranspose):
    def __init__(
        self,
        in_features,
        out_features,
        init_norm=1.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.register_buffer("u", self._generate_random_input())
        norm = self.compute_spectral_norm(k=1000)
        with torch.no_grad():
            self.weight *= init_norm / norm
        self._norm = init_norm

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def _generate_random_input(self) -> Tensor:
        return torch.randn(
            1, self.weight.shape[1], device=self.weight.device, dtype=self.weight.dtype
        )

    def tangent(self, x: Tensor) -> Tensor:
        return x.mm(self.weight.T)

    def transpose(self, z: Tensor) -> Tensor:
        return z.mm(self.weight)


class Conv1dNormalised(Conv1d, ModuleTranspose):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_size=(128,),
        init_norm=1.0,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self._input_size = input_size
        self.register_buffer("u", self._generate_random_input())
        norm = self.compute_spectral_norm(k=1000)
        with torch.no_grad():
            self.weight *= init_norm / norm
        self._norm = init_norm

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def _generate_random_input(self) -> Tensor:
        return torch.randn(
            1,
            self.in_channels,
            *self.input_size,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, val: Tuple[int]):
        if val != self.input_size:
            warn(f"Input size has changed from {self.input_size} to {val}.")
        self._input_size = val

    @property
    def _output_padding(self):
        output_padding = ()
        for inp_, ker_, str_, pad_, dil_ in zip(
            self.input_size, self.kernel_size, self.stride, self.padding, self.dilation
        ):
            output_padding += ((inp_ + 2 * pad_ - dil_ * (ker_ - 1) - 1) % str_,)
        return output_padding

    def tangent(self, x: Tensor) -> Tensor:
        return self._conv_forward(x, self.weight, None)

    def transpose(self, z: Tensor) -> Tensor:
        if sum(self.padding) > 0 and self.padding_mode != "zeros":
            raise NotImplementedError(
                "Transpose Conv1d not implemented yet for non-zero padding"
            )
        return F.conv_transpose1d(
            z,
            self.weight,
            None,
            self.stride,
            self.padding,
            self._output_padding,
            self.groups,
            self.dilation,
        )


class Conv2dNormalised(Conv2d, ModuleTranspose):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_size=(321, 481),
        init_norm=1.0,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self._input_size = input_size
        self.register_buffer("u", self._generate_random_input())
        norm = self.compute_spectral_norm(k=1000)
        with torch.no_grad():
            self.weight *= init_norm / norm
        self._norm = init_norm

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def _generate_random_input(self) -> Tensor:
        return torch.randn(
            1,
            self.in_channels,
            *self.input_size,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, val: Tuple[int, int]):
        if val != self.input_size:
            warn(f"Input size has changed from {self.input_size} to {val}.")
        self._input_size = val

    @property
    def _output_padding(self):
        output_padding = ()
        for inp_, ker_, str_, pad_, dil_ in zip(
            self.input_size, self.kernel_size, self.stride, self.padding, self.dilation
        ):
            output_padding += ((inp_ + 2 * pad_ - dil_ * (ker_ - 1) - 1) % str_,)
        return output_padding

    def tangent(self, x: Tensor) -> Tensor:
        return self._conv_forward(x, self.weight, None)

    def transpose(self, z: Tensor) -> Tensor:
        if sum(self.padding) > 0 and self.padding_mode != "zeros":
            raise NotImplementedError(
                "Transpose Conv2d not implemented yet for non-zero padding"
            )
        return F.conv_transpose2d(
            z,
            self.weight,
            None,
            self.stride,
            self.padding,
            self._output_padding,
            self.groups,
            self.dilation,
        )


class Conv3dNormalised(Conv3d, ModuleTranspose):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_size=(32, 32, 32),
        init_norm=1.0,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self._input_size = input_size
        self.register_buffer("u", self._generate_random_input())
        norm = self.compute_spectral_norm(k=1000)
        with torch.no_grad():
            self.weight *= init_norm / norm
        self._norm = init_norm

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def _generate_random_input(self) -> Tensor:
        return torch.randn(
            1,
            self.in_channels,
            *self.input_size,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, val: Tuple[int, int]):
        if val != self.input_size:
            warn(f"Input size has changed from {self.input_size} to {val}.")
        self._input_size = val

    @property
    def _output_padding(self):
        output_padding = ()
        for inp_, ker_, str_, pad_, dil_ in zip(
            self.input_size, self.kernel_size, self.stride, self.padding, self.dilation
        ):
            output_padding += ((inp_ + 2 * pad_ - dil_ * (ker_ - 1) - 1) % str_,)
        return output_padding

    def tangent(self, x: Tensor) -> Tensor:
        return self._conv_forward(x, self.weight, None)

    def transpose(self, z: Tensor) -> Tensor:
        if sum(self.padding) > 0 and self.padding_mode != "zeros":
            raise NotImplementedError(
                "Transpose Conv3d not implemented yet for non-zero padding"
            )
        return F.conv_transpose3d(
            z,
            self.weight,
            None,
            self.stride,
            self.padding,
            self._output_padding,
            self.groups,
            self.dilation,
        )
