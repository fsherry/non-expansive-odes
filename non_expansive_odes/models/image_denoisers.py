import torch

from .activation_functions import LeakyReLU
from .blocks import NonexpansiveBlock
from .integrators import euler
from .linear import Conv2dNormalised, ModuleTranspose


class DnCNN(torch.nn.Module):
    def __init__(
        self,
        in_channels=3,
        kernel_size=(3, 3),
        int_channels=64,
        blocks=20,
        residual=True,
        device=None,
    ):
        super().__init__()
        self._residual = residual
        self.lift = torch.nn.Conv2d(
            in_channels,
            int_channels,
            kernel_size,
            padding=tuple(k // 2 for k in kernel_size),
            device=device,
        )
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    int_channels,
                    int_channels,
                    kernel_size,
                    padding=tuple(k // 2 for k in kernel_size),
                    device=device,
                )
                for _ in range(blocks - 2)
            ]
        )
        self.bns = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm2d(int_channels, device=device)
                for _ in range(blocks - 2)
            ]
        )
        self.project = torch.nn.Conv2d(
            int_channels,
            in_channels,
            kernel_size,
            padding=tuple(k // 2 for k in kernel_size),
            device=device,
        )

    def forward(self, x):
        z = torch.nn.functional.relu(self.lift(x))
        for conv, bn in zip(self.convs, self.bns):
            z = torch.nn.functional.relu(bn(conv(z)))
        if self._residual:
            return x - self.project(z)
        else:
            return self.project(z)


class NonexpansiveDenoiser(torch.nn.Module):
    def __init__(
        self,
        in_channels=3,
        kernel_size=(3, 3),
        input_size=(321, 481),
        int_channels=64,
        blocks=10,
        residual=True,
        integrator=None,
        device=None,
    ):
        super().__init__()
        integrator = euler if integrator is None else integrator
        self._in_channels = in_channels
        self._int_channels = int_channels
        self._residual = residual
        self.blocks = torch.nn.Sequential(
            *[
                NonexpansiveBlock(
                    Conv2dNormalised(
                        int_channels,
                        int_channels,
                        kernel_size,
                        input_size,
                        padding=tuple(k // 2 for k in kernel_size),
                        device=device,
                    ),
                    LeakyReLU(),
                    integrator,
                )
                for _ in range(blocks)
            ]
        )

    def compute_spectral_norms(self, k: int = 1) -> None:
        for submodule in self.modules():
            if isinstance(submodule, ModuleTranspose):
                submodule.compute_spectral_norm(k=k)

    def forward(self, x):
        z = torch.cat(
            (
                x,
                torch.zeros(
                    (x.shape[0], self._int_channels - self._in_channels, *x.shape[2:]),
                    dtype=x.dtype,
                    device=x.device,
                ),
            ),
            dim=1,
        )
        z = self.blocks(z)[:, : self._in_channels]
        if self._residual:
            return x - z
        else:
            return z
