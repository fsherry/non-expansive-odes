import os
from argparse import ArgumentParser

import kornia
import torch
import wandb
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR

from non_expansive_odes.models.image_classifiers import ResNetClassifier
from non_expansive_odes.utils.cifar10_data import get_cifar10_splits
from non_expansive_odes.utils.classification import (
    attack,
    certified_robustness,
    robustness_auc,
    training_step,
    validate,
)
from non_expansive_odes.utils.loggers import log_gradient_norms

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", default=".", type=str)
    parser.add_argument("--results_path", default=".", type=str)
    parser.add_argument("--name", default="cifar10_nonexpansive", type=str)
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--max_lr", default=1, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--steps", default=400 * 100, type=int)
    parser.add_argument("--test_lr", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--margin", default=1.0, type=float)
    parser.add_argument("--int_channels", default=16, type=int)
    parser.add_argument("--blocks_per_stage", default=5, type=int)
    parser.add_argument("--stages", default=3, type=int)
    parser.add_argument("--factor", default=2, type=int)
    parser.add_argument("--device", default=0, type=int)
    args = parser.parse_args()

    params = vars(args)

    train_loader, val_loader, test_loader = get_cifar10_splits(
        batch_size=params["batch_size"],
        device=params["device"],
        data_path=params["data_path"],
        half=params["fp16"],
    )

    aug = torch.nn.Sequential(
        kornia.augmentation.RandomErasing((0.05, 0.2)),
        kornia.augmentation.RandomCrop((32, 32), (2, 2)),
        kornia.augmentation.RandomHorizontalFlip(),
    )

    run = wandb.init(
        project="nonexpansive-odes",
        group="adversarial-robustness-cifar10",
        name=params["name"] if not params["test_lr"] else params["name"] + "-test_lr",
        dir=os.getenv("WANDB_DIR"),
        config=params,
    )
    if not os.path.isdir(os.path.join(params["results_path"], run.group)):
        os.mkdir(os.path.join(params["results_path"], run.group))

    model = ResNetClassifier(
        int_channels=params["int_channels"],
        stages=params["stages"],
        blocks_per_stage=params["blocks_per_stage"],
        factor=params["factor"],
        device=params["device"],
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
            pct_start=0.2,
            anneal_strategy="linear",
        )
    iterate, end = 0, False
    best_val_acc = 0.0
    epsilons = torch.linspace(0, 1, 40)
    while True:
        for x, y in train_loader:
            batch = (x, y)
            loss, acc = training_step(model, optimiser, scheduler, params, batch, aug)
            iterate += 1
            log_gradient_norms(model, run, step=iterate)
            wandb.log(
                {
                    "train/loss": loss,
                    "train/acc": acc,
                    "train/learning_rate": optimiser.param_groups[0]["lr"],
                },
                step=iterate,
            )

            if iterate >= params["steps"]:
                end = True
                break
        if not params["test_lr"]:
            loss, acc = validate(model, val_loader, params)
            certified_robust = certified_robustness(model, val_loader, epsilons)
            auc = robustness_auc(certified_robust, epsilons)
            wandb.log(
                {
                    "validate/loss": loss,
                    "validate/acc": acc,
                    "validate/certified_robustness_auc": auc,
                    "validate/certified_robustness0.15": certified_robust[6],
                },
                step=iterate,
            )
            if best_val_acc < acc:
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
        loss, acc = validate(model, test_loader, params)
        wandb.log({"test/loss": loss, "test/acc": acc}, step=iterate)
        certified_robust = certified_robustness(model, test_loader, epsilons)
        adversarially_robust = attack(model, test_loader, epsilons)
        auc = robustness_auc(adversarially_robust, epsilons)
        wandb.log({"test/robustness_auc": auc}, step=iterate)
        torch.save(
            {
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epsilons": epsilons,
                "certified_robust": certified_robust,
                "adversarially_robust": adversarially_robust,
            },
            os.path.join(params["results_path"], run.group, f"{run.name}_best_model"),
        )
        run.summary["epsilons"] = epsilons
        run.summary["certified_robust"] = certified_robust
        run.summary["adversarially_robust"] = adversarially_robust
    run.finish()
