from typing import Callable

from torch import Tensor


def euler(
    xinit: Tensor, rhs: Callable[[Tensor], Tensor], step_size: float, n_steps: int
) -> Tensor:
    x = xinit
    for _ in range(n_steps):
        x = x + step_size * rhs(x)
    return x


euler.r_cc = 1.0


def heun(
    xinit: Tensor, rhs: Callable[[Tensor], Tensor], step_size: float, n_steps: int
) -> Tensor:
    x = xinit
    for _ in range(n_steps):
        x1 = rhs(x)
        x = x + 0.5 * step_size * (x1 + rhs(x + step_size * x1))
    return x


heun.r_cc = 1.0


def rk4(
    xinit: Tensor, rhs: Callable[[Tensor], Tensor], step_size: float, n_steps: int
) -> Tensor:
    x = xinit
    for _ in range(n_steps):
        x1 = rhs(x)
        x2 = rhs(x + 0.5 * step_size * x1)
        x3 = rhs(x + 0.5 * step_size * x2)
        x4 = rhs(x + step_size * x3)
        x = x + step_size * (x1 + 2 * x2 + 2 * x3 + x4) / 6
    return x


rk4.r_cc = 1.0
