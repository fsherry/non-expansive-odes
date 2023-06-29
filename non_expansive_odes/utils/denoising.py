import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from non_expansive_odes.models.image_denoisers import NonexpansiveDenoiser


def get_mses(xhat: Tensor, x: Tensor) -> Tensor:
    return torch.mean(
        (xhat.view(xhat.shape[0], -1) - x.view(x.shape[0], -1)) ** 2, dim=1
    )


def get_psnrs(xhat: Tensor, x: Tensor) -> Tensor:
    mses = get_mses(xhat, x)
    return 10 * torch.log10(torch.max(x.view(x.shape[0], -1) ** 2, dim=1).values / mses)


def training_step(model: Module, optimiser, scheduler, batch, aug=None, k=1):
    model.train()
    if isinstance(model, NonexpansiveDenoiser):
        model.compute_spectral_norms(k)
    optimiser.zero_grad()
    x, noisy = batch
    if aug is not None:
        noisy = aug(noisy)
    xhat = model(noisy)
    loss = torch.mean((xhat - x) ** 2)
    loss.backward()
    with torch.no_grad():
        psnrs = get_psnrs(xhat, x)
    optimiser.step()
    scheduler.step()
    return loss.item(), psnrs


def validate(model: Module, dataloader: DataLoader, k=1000):
    model.eval()
    if isinstance(model, NonexpansiveDenoiser):
        model.compute_spectral_norms(k)
    loss = 0.0
    psnrs = []
    N = 0
    for x, noisy in dataloader:
        with torch.no_grad():
            xhat = model(noisy)
            loss += noisy.shape[0] * torch.mean((xhat - x) ** 2)
            psnrs.append(get_psnrs(xhat, x))
            N += noisy.shape[0]
    return loss / N, torch.cat(psnrs)
