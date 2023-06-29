import torch
from torch.nn import AvgPool2d, Module, ModuleList, Sequential

from .activation_functions import LeakyReLU
from .blocks import NonexpansiveBlock, ResidualBlock
from .linear import Conv2d, Conv2dNormalised, Linear, LinearNormalised, ModuleTranspose


class NonexpansiveClassifier(Module):
    def __init__(
        self,
        in_channels=3,
        n_classes=10,
        kernel_size=(3, 3),
        input_size=(32, 32),
        int_channels=16,
        blocks_per_stage=5,
        stages=3,
        factor=2,
        device=None,
    ):
        super().__init__()

        self.lift = Conv2dNormalised(
            in_channels, int_channels, (1, 1), input_size, device=device
        )
        self.input_sizes = [
            tuple(k // 2**i for k in input_size) for i in range(stages)
        ]
        self.blocks = ModuleList()
        for i in range(stages):
            self.blocks.append(
                Sequential(
                    *[
                        NonexpansiveBlock(
                            Conv2dNormalised(
                                factor**i * int_channels,
                                factor**i * int_channels,
                                kernel_size,
                                input_size=self.input_sizes[i],
                                padding=tuple(k // 2 for k in kernel_size),
                                device=device,
                            ),
                            LeakyReLU(),
                        )
                        for _ in range(blocks_per_stage)
                    ]
                )
            )
        self.linears = ModuleList()
        for i in range(stages):
            self.linears.append(
                Conv2dNormalised(
                    factor**i * int_channels,
                    factor ** (i + 1) * int_channels,
                    (1, 1),
                    input_size=self.input_sizes[i],
                    device=device,
                )
            )

        self.project = LinearNormalised(
            factor**stages * int_channels, n_classes, device=device
        )

    def compute_spectral_norms(self, k: int = 1) -> None:
        for submodule in self.modules():
            if isinstance(submodule, ModuleTranspose):
                submodule.compute_spectral_norm(k=k)

    def forward(self, x):
        z = self.lift(x)
        for block, linear in zip(self.blocks, self.linears):
            z = block(z)
            z = linear(z)
            z = torch.nn.functional.avg_pool2d(z, (2, 2))
        z = torch.mean(z, dim=(2, 3))
        return self.project(z)

    @property
    def lipschitz_const(self) -> float:
        L = self.lift.norm
        for linear in self.linears:
            L * linear.norm
        return L * self.project.norm


class ResNetClassifier(Module):
    def __init__(
        self,
        in_channels=3,
        n_classes=10,
        kernel_size=(3, 3),
        int_channels=16,
        blocks_per_stage=5,
        stages=3,
        factor=2,
        device=None,
    ):
        super().__init__()
        self.lift = Conv2d(in_channels, int_channels, (1, 1), device=device)
        self.blocks = ModuleList()
        for i in range(stages):
            self.blocks.append(
                Sequential(
                    *[
                        ResidualBlock(
                            Conv2d(
                                factor**i * int_channels,
                                factor**i * int_channels,
                                kernel_size,
                                padding=tuple(k // 2 for k in kernel_size),
                                device=device,
                            ),
                            Conv2d(
                                factor**i * int_channels,
                                factor**i * int_channels,
                                kernel_size,
                                padding=tuple(k // 2 for k in kernel_size),
                                device=device,
                            ),
                            LeakyReLU(),
                        )
                        for _ in range(blocks_per_stage)
                    ],
                    Conv2d(
                        factor**i * int_channels,
                        factor ** (i + 1) * int_channels,
                        (1, 1),
                        device=device,
                    ),
                    AvgPool2d((2, 2)),
                )
            )
            self.project = Linear(
                factor**stages * int_channels, n_classes, device=device
            )

    def forward(self, x):
        z = self.lift(x)
        for block in self.blocks:
            z = block(z)
        z = torch.mean(z, dim=(2, 3))
        return self.project(z)
