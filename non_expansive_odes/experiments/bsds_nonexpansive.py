import os
from argparse import ArgumentParser
from math import inf

import torch
import wandb
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR

from non_expansive_odes.models.image_denoisers import NonexpansiveDenoiser
from non_expansive_odes.models.integrators import euler, heun, rk4
from non_expansive_odes.utils.bsds_data import get_bsds_splits
from non_expansive_odes.utils.denoising import training_step, validate
from non_expansive_odes.utils.loggers import log_dynamic_blocks, log_gradient_norms

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", default=".", type=str)
    parser.add_argument("--results_path", default=".", type=str)
    parser.add_argument("--name", default="cifar10_nonexpansive", type=str)
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--max_lr", default=1, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--steps", default=400 * 100, type=int)
    parser.add_argument("--noise_level", default=0.15, type=float)
    parser.add_argument("--test_lr", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--integrator", default="euler", type=str)
    parser.add_argument("--int_channels", default=64, type=int)
    parser.add_argument("--blocks", default=10, type=int)
    parser.add_argument("--device", default=0, type=int)
    args = parser.parse_args()

    params = vars(args)

    if params["integrator"] == "euler":
        integrator = euler
    elif params["integrator"] == "heun":
        integrator = heun
    elif params["integrator"] == "rk4":
        integrator = rk4
    else:
        raise NotImplementedError(
            f"Integrator {params['integrator']} is not implemented."
        )

    train_loader, val_loader, test_loader = get_bsds_splits(
        batch_size=params["batch_size"],
        device=params["device"],
        data_path=params["data_path"],
        noise_level=params["noise_level"],
        half=params["fp16"],
    )

    run = wandb.init(
        project="nonexpansive-odes",
        group="gaussian-denoising-bsds500",
        name=params["name"] if not params["test_lr"] else params["name"] + "-test_lr",
        dir=os.getenv("WANDB_DIR"),
        config=params,
    )

    if not os.path.isdir(os.path.join(params["results_path"], run.group)):
        os.mkdir(os.path.join(params["results_path"], run.group))

    model = NonexpansiveDenoiser(
        int_channels=params["int_channels"],
        blocks=params["blocks"],
        device=params["device"],
        residual=params["residual"],
        integrator=integrator,
    )
    if params["fp16"]:
        model = model.half()

    optimiser = SGD(
        model.parameters(), lr=params["max_lr"], momentum=0.9, weight_decay=params["wd"]
    )
    if params["test_lr"]:
        scheduler = OneCycleLR(
            optimiser,
            max_lr=params["max_lr"],
            div_factor=params["max_lr"] / params["min_lr"],
            total_steps=params["steps"],
            pct_start=1.0,
            anneal_strategy="cos",
        )
    else:
        scheduler = OneCycleLR(
            optimiser,
            max_lr=params["max_lr"],
            div_factor=params["max_lr"] / params["min_lr"],
            total_steps=params["steps"],
            pct_start=0.5,
            anneal_strategy="linear",
        )
    iterate, end = 0, False
    best_val_psnr = -inf
    while True:
        for batch in train_loader:
            loss, psnrs = training_step(model, optimiser, scheduler, batch)
            iterate += 1
            log_gradient_norms(model, run, iterate)
            log_dynamic_blocks(model, run, iterate)
            wandb.log(
                {
                    "train/loss": loss,
                    "train/psnr_mean": psnrs.mean().item(),
                    "train/psnr_std": psnrs.std().item(),
                    "train/learning_rate": optimiser.param_groups[0]["lr"],
                },
                step=iterate,
            )

            if iterate >= params["steps"]:
                end = True
                break
        if not params["test_lr"]:
            loss, psnrs = validate(model, val_loader)
            wandb.log(
                {
                    "validate/loss": loss,
                    "validate/psnrs_mean": psnrs.mean().item(),
                    "validate/psnrs_std": psnrs.std().item(),
                },
                step=iterate,
            )
            if best_val_psnr < psnrs.mean():
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimiser": optimiser.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    os.path.join(
                        params["results_path"], run.group, f"{run.name}_best_model"
                    ),
                )
        if end:
            break

    if not params["test_lr"]:
        model.load_state_dict(
            torch.load(
                os.path.join(
                    params["results_path"], run.group, f"{run.name}_best_model"
                )
            )["model"]
        )
        loss, psnrs = validate(model, test_loader)
        wandb.log(
            {
                "test/loss": loss,
                "test/psnrs_mean": psnrs.mean().item(),
                "test/psnrs_std": psnrs.std().item(),
            }
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "psnrs": psnrs,
            },
            os.path.join(params["results_path"], run.group, f"{run.name}_best_model"),
        )
    run.finish()
