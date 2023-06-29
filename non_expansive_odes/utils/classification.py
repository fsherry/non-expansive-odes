from math import sqrt

import foolbox
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from non_expansive_odes.models.image_classifiers import NonexpansiveClassifier


def get_margins(model: Module, x: Tensor, y: Tensor):
    model.eval()
    with torch.no_grad():
        preds = model(x)
    score_correct = preds.gather(1, y.view(-1, 1)).squeeze()
    margins = (
        score_correct
        - preds.gather(
            1,
            torch.ones_like(preds)
            .scatter(1, y.view(-1, 1), 0)
            .nonzero()[:, 1]
            .reshape(-1, 9),
        )
        .max(dim=1)
        .values
    )
    return margins


def certified_robustness(
    model: Module,
    dataloader: DataLoader,
    epsilons: Tensor,
    lipschitz_const: float = 1.0,
):
    device = next(iter(model.parameters())).device
    N = 0
    robust = torch.zeros(epsilons.shape, device=device)
    for x, y in dataloader:
        margins = get_margins(model, x, y)
        robust += (
            margins.view(-1, 1)
            > sqrt(2) * lipschitz_const * epsilons.view(1, -1).to(device)
        ).sum(dim=0)
        N += x.shape[0]
    return robust / N


def attack(
    model: Module,
    dataloader: DataLoader,
    epsilons: Tensor,
    n_iter: int = 10,
):
    model.eval()
    device = next(iter(model.parameters())).device
    model_fb = foolbox.PyTorchModel(model, (0, 1), device=device)
    attack = foolbox.attacks.L2PGD(steps=n_iter)
    N = 0
    robust = torch.zeros(epsilons.shape, device=device)
    for x, y in dataloader:
        _, _, res = attack(model_fb, x, y, epsilons=epsilons)
        robust += x.shape[0] - res.sum(dim=1)
        N += x.shape[0]
    return robust / N


def robustness_auc(robustness: Tensor, epsilons: Tensor):
    # Compute an AUC for the robust accuracy using the trapezoidal method
    robustness_aves = torch.nn.functional.conv1d(
        robustness.view(1, 1, -1),
        torch.tensor([[[0.5, 0.5]]], device=robustness.device),
    ).view(-1)
    delta_epsilons = epsilons.to(robustness.device).diff()
    return (robustness_aves * delta_epsilons).sum().item()


def training_step(model: Module, optimiser, scheduler, params, batch, aug=None, k=1):
    model.train()
    if isinstance(model, NonexpansiveClassifier):
        model.compute_spectral_norms(k)
    optimiser.zero_grad()
    x, y = batch
    if aug is not None:
        x = aug(x)
    yhat = model(x)
    with torch.no_grad():
        acc = torch.mean(1.0 * (torch.argmax(yhat, dim=1) == y))
    loss = torch.nn.functional.cross_entropy(yhat, y)
    #    loss = torch.nn.functional.multi_margin_loss(yhat, y, margin=params["margin"])
    loss.backward()
    optimiser.step()
    scheduler.step()
    # if isinstance(model, NonexpansiveClassifier):
    #     model.lift.rescale()
    #     for linear in model.linears:
    #         linear.rescale()
    #     model.project.rescale()
    return loss.item(), acc.item()


def validate(model: Module, dataloader: DataLoader, params, k=1000):
    model.eval()
    if isinstance(model, NonexpansiveClassifier):
        model.compute_spectral_norms(k)
    loss = 0.0
    correct = 0.0
    N = 0
    for x, y in dataloader:
        x, y = x.to(params["device"]), y.to(params["device"])
        with torch.no_grad():
            yhat = model(x)
            correct += torch.sum(1.0 * (torch.argmax(yhat, dim=1) == y)).item()
            N += yhat.shape[0]
            # loss += torch.nn.functional.multi_margin_loss(
            #     yhat, y, margin=params["margin"], reduction="sum"
            # ).item()
            loss += torch.nn.functional.cross_entropy(yhat, y).item()
    return loss / N, correct / N
