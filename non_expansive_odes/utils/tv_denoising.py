from math import sqrt

import torch


def grad(x):
    grad_1 = torch.cat(
        (
            x[:, :, :-1, :] - x[:, :, 1:, :],
            torch.zeros(*x.shape[:2], 1, x.shape[3], device=x.device),
        ),
        dim=2,
    )
    grad_2 = torch.cat(
        (
            x[:, :, :, :-1] - x[:, :, :, 1:],
            torch.zeros(*x.shape[:2], x.shape[2], 1, device=x.device),
        ),
        dim=3,
    )
    return grad_1, grad_2


def div(x):
    x1, x2 = x
    diff_x = torch.cat(
        (-x1[:, :, 0:1, :], x1[:, :, :-2, :] - x1[:, :, 1:-1, :], x1[:, :, -2:-1, :]),
        dim=2,
    )
    diff_y = torch.cat(
        (-x2[:, :, :, 0:1], x2[:, :, :, :-2] - x2[:, :, :, 1:-1], x2[:, :, :, -2:-1]),
        dim=3,
    )
    return -(diff_x + diff_y)


def proxG(x, eps, tau):
    return x / (1 + eps * tau)


def proxF1(z, y, tau):
    return (z + tau * y) / (tau + 1)


def proxF1star(z, y, tau):
    return z - tau * proxF1(z / tau, y, 1.0 / tau)


def proxF2star(z, alpha, tau):
    z1, z2 = z
    sqr_norm = z1**2 + z2**2
    ind = sqr_norm > alpha**2
    z1[ind] *= alpha / torch.sqrt(sqr_norm[ind])
    z2[ind] *= alpha / torch.sqrt(sqr_norm[ind])
    return (z1, z2)


def proxFstar(z, y, alpha, tau):
    z1, z2 = z
    return (proxF1star(z1, y, tau), proxF2star(z2, alpha, tau))


def tv_denoising(y, alpha=1e-4, it=1000, eps=1e-8):
    """Implements the Chambolle-Pock primal-dual hybrid gradient method to
    solve the variational problem
    min_x ||x - y||_2^2/2 + alpha ||grad x||_1 + eps ||x||_2^2
    """
    op_norm = 3
    x = y.clone()
    xold = x.clone()
    xbar = x.clone()
    z = x, grad(x)
    tau, sigma = 1.0 / op_norm, 1.0 / op_norm
    for i in range(it):
        z1, z2 = z
        K1, K2 = xbar, grad(xbar)
        z = proxFstar(
            (z1 + sigma * K1, tuple(z_ + sigma * K_ for z_, K_ in zip(z2, K2))),
            y,
            alpha,
            tau,
        )
        x = proxG(xold - tau * (z[0] + div(z[1])), eps, tau)
        theta = 1.0 / sqrt(1 + 2 * eps * tau)
        tau *= theta
        sigma /= theta
        xbar = x + theta * (x - xold)
        xold = x
    return x
