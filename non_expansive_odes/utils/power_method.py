from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.nn import Module


def power_method_nonsquare(
    A: Callable[[Tensor], Tensor],
    A_t: Callable[[Tensor], Tensor],
    u_init: Tensor,
    k: int = 1,
) -> Tuple[float, Tensor]:
    """`power_method_nonsquare` computes the first singular value (operator norm) of a
    linear operator.

    The return values are the first singular value and its right singular vector.
    """
    u = u_init
    u /= torch.sqrt(torch.sum(u**2))
    for i in range(k):
        v = A(u)
        u = A_t(v)
        norm_ATAu = torch.norm(u)
        u /= norm_ATAu
    return torch.sqrt(norm_ATAu).item(), u


def power_method_square(
    A: Callable[[Tensor], Tensor], u_init: Tensor, k: int = 1
) -> Tuple[float, Tensor]:
    """`power_method_square` computes the top eigenvalue of a square linear operator.

    The return values are the first eigenvalue and corresponding eigenvector.

    N.B.: The eigenvalue is not generally the operator norm. If the operator norm is
    wanted, the user should ensure that the operator is symmetric, say, or use the
    function `power_method_nonsquare`

    """
    u = u_init
    u /= torch.norm(u)
    for i in range(k):
        u = A(u)
        norm_Au = torch.norm(u)
        u /= norm_Au
    return norm_Au.item(), u


def power_method_jacobian(
    Phi: Module, x: Tensor, u_init: Tensor, k: int = 1
) -> Tuple[float, Tensor]:
    """`power_method_jacobian` applies the power method to the Jacobian of a PyTorch
    Module.

    The return values are the first eigenvalue and corresponding eigenvector.

    N.B.: This eigenvalue is not generally the operator norm. If the operator norm is
    wanted, the user should ensure that the Jacobian is symmetric, say. An alternative
    approach that has not been implemented here is to apply the methodology of
    `power_method_nonsquare`, which requires forward mode AD.
    """

    def A(u: Tensor) -> Tensor:
        x.requires_grad = True
        x.grad = None if x.grad is not None else x.grad
        Phi.zero_grad()
        torch.sum(u * Phi(x)).backward()
        return x.grad.clone().detach()

    return power_method_square(A, u_init, k)
