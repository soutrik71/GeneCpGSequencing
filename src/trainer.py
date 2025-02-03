import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, List
import time
from torch.amp import GradScaler, autocast
from src.utils import save_checkpoint, load_checkpoint


def training_loop_pack_padded(
    model, dataloader, device, optimizer, criterion, grad_scaler
):
    """
    Runs one training epoch.

    Parameters:
    - model (nn.Module): PyTorch model to train.
    - dataloader (DataLoader): Training DataLoader.
    - device (torch.device): CPU or GPU.
    - optimizer (torch.optim.Optimizer): Optimizer.
    - criterion (nn.Module): Loss function.
    - grad_scaler (GradScaler): Gradient scaler for mixed precision training.

    Returns:
    - avg_loss (float): Average loss over dataset.
    - avg_mae (float): Average MAE over dataset.
    - avg_rmse (float): Average RMSE over dataset.
    """
    model.train()
    total_loss, total_mae, total_rmse = 0.0, 0.0, 0.0
    num_batches = len(dataloader)

    for inputs, labels, lengths in dataloader:
        inputs, labels, lengths = (
            inputs.to(device),
            labels.to(device),
            lengths.to(device),
        )

        optimizer.zero_grad()

        with autocast("cuda" if device.type == "cuda" else "cpu"):
            outputs = model(inputs, lengths).squeeze()
            loss = criterion(outputs, labels)

        grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        # Compute batch-wise metrics
        batch_mae = mean_absolute_error(
            labels.cpu().numpy(), outputs.detach().cpu().numpy()
        )
        batch_rmse = (
            mean_squared_error(labels.cpu().numpy(), outputs.detach().cpu().numpy())
            ** 0.5
        )

        total_loss += loss.item()
        total_mae += batch_mae
        total_rmse += batch_rmse

    return total_loss / num_batches, total_mae / num_batches, total_rmse / num_batches


def validation_loop_pack_padded(model, dataloader, device, criterion):
    """
    Runs one validation epoch.

    Parameters:
    - model (nn.Module): PyTorch model to evaluate.
    - dataloader (DataLoader): Validation DataLoader.
    - device (torch.device): CPU or GPU.
    - criterion (nn.Module): Loss function.

    Returns:
    - avg_loss (float): Average loss over dataset.
    - avg_mae (float): Average MAE over dataset.
    - avg_rmse (float): Average RMSE over dataset.
    """
    model.eval()
    total_loss, total_mae, total_rmse = 0.0, 0.0, 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for inputs, labels, lengths in dataloader:
            inputs, labels, lengths = (
                inputs.to(device),
                labels.to(device),
                lengths.to(device),
            )
            outputs = model(inputs, lengths).squeeze()

            loss = criterion(outputs, labels)

            # Compute batch-wise metrics
            batch_mae = mean_absolute_error(labels.cpu().numpy(), outputs.cpu().numpy())
            batch_rmse = (
                mean_squared_error(labels.cpu().numpy(), outputs.cpu().numpy()) ** 0.5
            )

            total_loss += loss.item()
            total_mae += batch_mae
            total_rmse += batch_rmse

    return total_loss / num_batches, total_mae / num_batches, total_rmse / num_batches


def train_model_pack_padded(
    model,
    train_loader,
    val_loader,
    device,
    epochs=25,
    patience=5,
    save_path="best_cpg_model_advanced.pth",
    lr=0.001,
    weight_decay=1e-4,
    resume=False,
):
    """
    Trains an LSTM model with validation and early stopping. Supports resuming training.

    Parameters:
    - model: LSTM model.
    - train_loader: Training DataLoader.
    - val_loader: Validation DataLoader.
    - device: CPU or GPU.
    - epochs: Max training epochs.
    - patience: Early stopping patience.
    - save_path: Path to save the best model.
    - lr: Initial learning rate.
    - weight_decay: L2 regularization weight.
    - resume: Whether to resume training from the last checkpoint.

    Returns:
    - Best trained model.
    """
    # initialize model and optimizer and scheduler and criterion and best_val_loss and start_epoch
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2,
    )
    # initialize gradient scaler
    grad_scaler = GradScaler()

    best_val_loss = float("inf")
    start_epoch = 0

    # Load checkpoint if resuming
    if resume and os.path.exists(save_path):
        try:
            model, optimizer, scheduler, start_epoch, best_val_loss = load_checkpoint(
                model, optimizer, scheduler, device, save_path
            )
        except FileNotFoundError:
            print("No checkpoint found. Starting training from scratch.")

    else:
        print("Starting training from scratch.")

    no_improvement = 0
    start_time = time.time()

    try:
        for epoch in range(start_epoch, epochs):
            train_loss, train_mae, train_rmse = training_loop_pack_padded(
                model, train_loader, device, optimizer, criterion, grad_scaler
            )
            val_loss, val_mae, val_rmse = validation_loop_pack_padded(
                model, val_loader, device, criterion
            )

            # Scheduler step
            if epoch > 0:
                scheduler.step(val_loss)

            # Print results
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}"
            )

            # Save best model checkpoint
            if val_loss < best_val_loss:
                print(
                    f"New best validation loss: {val_loss:.4f} (previous best: {best_val_loss:.4f})"
                )
                best_val_loss = val_loss
                no_improvement = 0
                save_checkpoint(
                    epoch, model, optimizer, scheduler, best_val_loss, save_path
                )
            else:
                no_improvement += 1
                print(f"No improvement, patience left: {patience - no_improvement}")

            # Early stopping
            if no_improvement >= patience:
                print(f"Early Stopping Triggered after epoch {epoch+1}")
                break

    except KeyboardInterrupt:
        print("Training Interrupted! Saving last checkpoint...")
        save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, save_path)

    print(f"Training Completed in {(time.time() - start_time):.2f} seconds")

    # Load the best model before returning
    model, optimizer, scheduler, _, _ = load_checkpoint(
        model, optimizer, scheduler, device, save_path
    )
    return model
