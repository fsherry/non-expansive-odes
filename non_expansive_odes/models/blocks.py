from math import ceil
from typing import Callable, Optional

from torch import Tensor
from torch.nn import Module

from .integrators import euler
from .linear import ModuleTranspose

Integrator = Callable[[Tensor, Callable[[Tensor], Tensor], float, int], Tensor]


class NonexpansiveBlock(Module):
    def __init__(
        self,
        linear: ModuleTranspose,
        nonlin: Module,
        integrator: Integrator = euler,
        timespan: Optional[float] = None,
        run_power_method=False,
    ):
        super().__init__()
        self.linear = linear
        if not hasattr(nonlin, "L"):
            raise AttributeError(
                "Nonlinearity requires a Lipschitz constant `L` as attribute"
            )
        if not hasattr(integrator, "r_cc"):
            raise AttributeError(
                "Integrator requires a circle contractivity radius `r_cc` as attribute"
            )
        self.nonlin = nonlin
        self.rhs = lambda x: -self.linear.transpose(self.nonlin(self.linear(x)))
        self.integrator = integrator
        if run_power_method:
            self.compute_spectral_norm(k=1000)
        self.timespan = self.r_cc / self.nu if timespan is None else timespan

    @property
    def step_size(self) -> float:
        return self.timespan / self.n_steps

    @property
    def n_steps(self) -> int:
        return ceil(self.timespan * self.nu / (2 * self.r_cc))

    @property
    def norm(self) -> float:
        return self.linear.norm

    @property
    def L(self) -> float:
        return self.nonlin.L

    @property
    def nu(self) -> float:
        return self.L * self.norm**2

    @property
    def r_cc(self) -> float:
        return self.integrator.r_cc

    def forward(self, x: Tensor) -> Tensor:
        return self.integrator(x, self.rhs, self.step_size, self.n_steps)

    def compute_spectral_norm(
        self, u_init: Optional[Tensor] = None, k: int = 1
    ) -> float:
        return self.linear.compute_spectral_norm(u_init, k)


class ResidualBlock(Module):
    def __init__(self, linearA: Module, linearB: Module, nonlin: Module):
        super().__init__()
        self.linearA = linearA
        self.linearB = linearB
        self.nonlin = nonlin

    def forward(self, x):
        return x - self.linearB(self.nonlin(self.linearA(x)))
