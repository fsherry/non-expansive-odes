from typing import Optional

import torch
from torch.nn import Module
from wandb.sdk.wandb_run import Run

from ..models.blocks import NonexpansiveBlock
from ..models.image_classifiers import NonexpansiveClassifier


def log_gradient_norms(model: Module, run: Run, step: Optional[int]):
    run.log(
        {
            "diagnostics/grad_norm_sqrd": sum(
                [
                    torch.sum(param.grad**2)
                    for param in model.parameters()
                    if param.grad
                ]
            ).item()
        },
        step=step,
    )


def log_dynamic_blocks(model: Module, run: Run, step: Optional[int]):
    stats = {}
    integrator_steps = 0
    n_blocks = 0
    for i, block in enumerate(
        filter(lambda module: isinstance(module, NonexpansiveBlock), model.modules())
    ):
        stats[f"diagnostics/block_norm_{i + 1}"] = block.norm
        stats[f"diagnostics/block_integrator_steps_{i + 1}"] = block.n_steps
        integrator_steps += block.n_steps
        n_blocks += 1
    stats["diagnostics/total_integrator_steps"] = integrator_steps
    stats["diagnostics/n_blocks"] = n_blocks
    run.log(stats, step=step)


def log_linear_layers(model: NonexpansiveClassifier, run: Run, step: Optional[int]):
    stats = {
        "diagnostics/lift_norm": model.lift.norm,
        "diagnostics/project_norm": model.project.norm,
    }
    for i, linear in enumerate(model.linears):
        stats[f"diagnostics/linear_norm_{i + 1}"] = linear.norm
    run.log(stats, step=step)
