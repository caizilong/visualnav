import wandb
import os
import json
import numpy as np
from typing import List, Optional, Dict
from prettytable import PrettyTable

from vint_train.training.train_utils import train, evaluate
from vint_train.training.train_utils import train_nomad, evaluate_nomad

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from vint_train.training.checkpoint import (
    make_nomad_checkpoint, restore_nomad_checkpoint, save_checkpoint, scheduler_state, unwrap_model,
)

def train_eval_loop(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    dataloader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    wandb_log_freq: int = 10,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        train_model: whether to train the model or not
        model: model to train
        optimizer: optimizer to use
        scheduler: learning rate scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        normalized: whether to normalize the action space or not
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        learn_angle: whether to learn the angle or not
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
    """
    assert 0 <= alpha <= 1
    latest_path = os.path.join(project_folder, f"latest.pth")

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start ViNT Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
            )

        avg_total_test_loss = []
        for dataset_type in test_dataloaders:
            print(
                f"Start {dataset_type} ViNT Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            loader = test_dataloaders[dataset_type]

            test_dist_loss, test_action_loss, total_eval_loss = evaluate(
                eval_type=dataset_type,
                model=model,
                dataloader=loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )

            avg_total_test_loss.append(total_eval_loss)

        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "scheduler": scheduler
        }
        # log average eval loss
        if use_wandb and wandb_log_freq > 0:
            wandb.log({}, commit=False)

        if scheduler is not None:
            # scheduler calls based on the type of scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(np.mean(avg_total_test_loss))
            else:
                scheduler.step()
        if use_wandb and wandb_log_freq > 0:
            wandb.log({
                "avg_total_test_loss": np.mean(avg_total_test_loss),
                "lr": optimizer.param_groups[0]["lr"],
            }, commit=False)

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, numbered_path)  # keep track of model at every epoch

    # Flush the last set of eval logs
    if use_wandb and wandb_log_freq > 0:
        wandb.log({})
    print()

def train_eval_loop_nomad(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam, 
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
    resume_checkpoint: Optional[dict] = None,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        model: model to train
        optimizer: optimizer to use
        lr_scheduler: learning rate scheduler to use
        noise_scheduler: noise scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        goal_mask_prob: probability of masking the goal token during training
        epochs: total target number of epochs, including epochs completed before resume
        device: device to train on
        project_folder: folder to save checkpoints and logs
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
        eval_freq: frequency of evaluation
    """
    latest_path = os.path.join(project_folder, "latest.pth")
    ema_model = EMAModel(model=model, power=0.75)
    if resume_checkpoint is not None:
        current_epoch = restore_nomad_checkpoint(
            resume_checkpoint, model, ema_model, optimizer, lr_scheduler,
        )
        print(f"Restored NoMaD at epoch {current_epoch}; EMA steps: {ema_model.optimization_step}")
    
    # epochs is the total target, including already completed epochs on resume.
    for epoch in range(current_epoch, epochs):
        metrics = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        train_metrics = {}
        if train_model:
            print(
            f"Start NoMaD Training Epoch {epoch}/{epochs - 1}"
            )
            train_metrics = train_nomad(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
            )
            metrics["train"] = train_metrics

        evaluation_losses = []
        if eval_freq > 0 and (epoch + 1) % eval_freq == 0:
            for dataset_type in test_dataloaders:
                print(
                    f"Start {dataset_type} NoMaD Testing Epoch {epoch}/{epochs - 1}"
                )
                loader = test_dataloaders[dataset_type]
                eval_metrics = evaluate_nomad(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    image_log_freq=image_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                )
                metrics[dataset_type] = eval_metrics
                evaluation_losses.append(eval_metrics["diffusion_eval_loss (random masking)"])

        if train_model and lr_scheduler is not None:
            monitor = float(np.mean(evaluation_losses)) if evaluation_losses else train_metrics["diffusion_loss"]
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(monitor)
            elif isinstance(getattr(lr_scheduler, "after_scheduler", None), torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(metrics=monitor)
            else:
                lr_scheduler.step()
        metrics["next_lr"] = optimizer.param_groups[0]["lr"]
        with open(os.path.join(project_folder, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(f"Epoch {epoch} metrics: {metrics}")

        if train_model:
            model_state = unwrap_model(model).state_dict()
            ema_state = unwrap_model(ema_model.averaged_model).state_dict()
            save_checkpoint(model_state, latest_path)
            save_checkpoint(model_state, os.path.join(project_folder, f"{epoch}.pth"))
            save_checkpoint(ema_state, os.path.join(project_folder, "ema_latest.pth"))
            save_checkpoint(ema_state, os.path.join(project_folder, f"ema_{epoch}.pth"))
            save_checkpoint(optimizer.state_dict(), os.path.join(project_folder, "optimizer_latest.pth"))
            if lr_scheduler is not None:
                # Full resume uses the recursively serialized scheduler in checkpoint_latest.
                save_checkpoint(scheduler_state(lr_scheduler), os.path.join(project_folder, "scheduler_latest.pth"))
            checkpoint = make_nomad_checkpoint(model, ema_model, optimizer, lr_scheduler, epoch)
            save_checkpoint(checkpoint, os.path.join(project_folder, "checkpoint_latest.pth"))
            print(f"Saved model, EMA and complete training checkpoint to {project_folder}")
        if use_wandb and wandb_log_freq > 0:
            wandb.log({"lr": metrics["lr"], "epoch": epoch})

    # Flush the last set of eval logs
    if use_wandb and wandb_log_freq > 0:
        wandb.log({})
    print()

def load_model(model, model_type, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    if model_type == "nomad":
        state_dict = checkpoint["model"] if "format_version" in checkpoint else checkpoint
        unwrap_model(model).load_state_dict(state_dict)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)


def load_ema_model(ema_model, state_dict: dict) -> None:
    """Load model from checkpoint."""
    ema_model.load_state_dict(state_dict)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    # print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params
