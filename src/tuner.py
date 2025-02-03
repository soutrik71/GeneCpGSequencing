# Define the objective function for Optuna
import optuna
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, List
import time
from torch.amp import GradScaler, autocast
from src.model import CpGCounterAdvancedPackPadding
from src.trainer import (
    training_loop_pack_padded,
    validation_loop_pack_padded,
    train_model_pack_padded,
)
from functools import partial


def objective(trial, vocab_size, train_dataloader, val_dataloader, device):
    """
    Objective function for Optuna hyperparameter optimization.

    Parameters:
    - trial: Optuna trial object.

    Returns:
    - Best validation loss for the trial.
    """

    # Sample hyperparameters
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.25)

    # Initialize the model and also optimizer, criterion, scheduler, and gradient scaler
    model = CpGCounterAdvancedPackPadding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    grad_scaler = GradScaler()

    # Training parameters
    num_epochs = 20
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss, _, _ = training_loop_pack_padded(
            model, train_dataloader, device, optimizer, criterion, grad_scaler
        )
        val_loss, _, _ = validation_loop_pack_padded(
            model, val_dataloader, device, criterion
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


# Optuna Study for Hyperparameter Optimization
def tune_hyperparameters(
    vocab_size,
    train_dataloader,
    val_dataloader,
    device,
    num_epochs,
    stop_patience,
    n_trials=10,
    save_best_model_path="best_cpg_model_optuna.pth",
    study_name="cpg_optuna",
):
    """
    Runs Optuna hyperparameter tuning and trains the best model.

    Parameters:
    - vocab_size (int): Number of unique tokens in the vocabulary.
    - train_dataloader (DataLoader): Training DataLoader.
    - val_dataloader (DataLoader): Validation DataLoader.
    - device (torch.device): CPU or GPU.
    - num_epochs (int): Max training epochs.
    - stop_patience (int): Early stopping patience.
    - n_trials (int): Number of Optuna trials.
    - save_best_model_path (str): Path to save the best model.
    - study_name (str): Name of the Optuna study.

    Returns:
    - Best hyperparameters.
    """

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(),
    )
    objective_with_args = partial(
        objective,
        vocab_size=vocab_size,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
    )
    study.optimize(objective_with_args, n_trials=n_trials)

    # Get the best hyperparameters
    best_params = study.best_params
    print(f"Best Hyperparameters: {best_params}")

    # Train model with best hyperparameters
    best_model = CpGCounterAdvancedPackPadding(
        vocab_size=vocab_size,
        embedding_dim=best_params["embedding_dim"],
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
    ).to(device)

    trained_model = train_model_pack_padded(
        best_model,
        train_dataloader,
        val_dataloader,
        device,
        epochs=num_epochs,
        patience=stop_patience,
        save_path=save_best_model_path,
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
    )

    return best_params, trained_model
