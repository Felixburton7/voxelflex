"""
Training command for Voxelflex.

This module handles the training of RMSF prediction models.
"""

import os
import time
import json
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from voxelflex.data.data_loader import load_voxel_data, load_rmsf_data, prepare_dataloaders
from voxelflex.models.cnn_models import get_model
from voxelflex.utils.logging_utils import get_logger, ProgressBar
from voxelflex.utils.file_utils import ensure_dir, save_json
from voxelflex.utils.system_utils import get_device

logger = get_logger(__name__)

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    show_progress: bool = True
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (CPU or GPU)
        epoch: Current epoch number
        show_progress: Whether to show progress bar
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    if show_progress:
        progress = ProgressBar(total=len(train_loader), prefix=f"Epoch {epoch+1}", suffix="Complete")
    
    for i, (inputs, targets) in enumerate(train_loader):
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        if show_progress:
            progress.update(i + 1)
    
    if show_progress:
        progress.finish()
    
    return running_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True
) -> float:
    """
    Validate model on validation set.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on (CPU or GPU)
        show_progress: Whether to show progress bar
        
    Returns:
        Average validation loss
    """
    model.eval()
    running_loss = 0.0
    
    if show_progress:
        progress = ProgressBar(total=len(val_loader), prefix="Validation", suffix="Complete")
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            
            if show_progress:
                progress.update(i + 1)
    
    if show_progress:
        progress.finish()
    
    return running_loss / len(val_loader)


def train_model(config: Dict[str, Any]) -> Tuple[str, Dict[str, List[float]]]:
    """
    Train an RMSF prediction model using the provided configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model_path, training_history)
    """
    # Get device (CPU or GPU)
    device = get_device(config["system_utilization"]["adjust_for_gpu"])
    logger.info(f"Using device: {device}")
    
    # Load data
    voxel_data = load_voxel_data(
        config["input"]["voxel_file"],
        config["input"]["domain_ids"] if config["input"]["domain_ids"] else None
    )
    
    rmsf_data = load_rmsf_data(
        config["input"]["rmsf_dir"],
        config["input"].get("replica", "replica_average"),  # Use get() with default
        config["input"]["temperature"]
    )
    
    # Prepare dataloaders
    num_workers = config.get("system_utilization", {}).get("num_workers", 4)
    train_loader, val_loader, test_loader = prepare_dataloaders(
        voxel_data,
        rmsf_data,
        batch_size=config["training"]["batch_size"],
        train_split=config["training"]["train_split"],
        val_split=config["training"]["val_split"],
        test_split=config["training"]["test_split"],
        num_workers=num_workers,
        seed=config.get("training", {}).get("seed", 42)
    )
    
    # Create model
    input_shape = None
    for domain_id, domain_data in voxel_data.items():
        for resid, voxel in domain_data.items():
            input_shape = voxel.shape
            break
        if input_shape is not None:
            break
    
    if input_shape is None:
        raise ValueError("Could not determine input shape from voxel data")
    
    logger.info(f"Input shape: {input_shape}")
    
    model = get_model(
        architecture=config["model"]["architecture"],
        input_channels=input_shape[0],
        channel_growth_rate=config["model"]["channel_growth_rate"],
        num_residual_blocks=config["model"]["num_residual_blocks"],
        dropout_rate=config["model"]["dropout_rate"],
        base_filters=config["model"]["base_filters"]
    )
    
    model = model.to(device)
    logger.info(f"Created {config['model']['architecture']} model")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    
    # Ensure numeric types for optimizer parameters
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create scheduler if specified
    scheduler = None
    if "scheduler" in config.get("training", {}):
        scheduler_config = config["training"]["scheduler"]
        if scheduler_config["type"] == "reduce_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get("mode", "min"),
                factor=float(scheduler_config.get("factor", 0.1)),
                patience=int(scheduler_config.get("patience", 10)),
                verbose=True
            )
        elif scheduler_config["type"] == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(scheduler_config.get("step_size", 30)),
                gamma=float(scheduler_config.get("gamma", 0.1))
            )
    
    # Train the model
    num_epochs = int(config["training"]["num_epochs"])
    show_progress = config["logging"]["show_progress_bars"]
    
    train_losses = []
    val_losses = []
    
    logger.info(f"Starting training for {num_epochs} epochs")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, show_progress
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, show_progress)
        val_losses.append(val_loss)
        
        # Update scheduler if used
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save the model
    model_dir = os.path.join(config["output"]["base_dir"], "models")
    ensure_dir(model_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"{config['model']['architecture']}_{timestamp}.pt")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'input_shape': input_shape,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epoch': num_epochs
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    
    history_path = os.path.join(model_dir, f"training_history_{timestamp}.json")
    save_json(history, history_path)
    
    logger.info(f"Training history saved to {history_path}")
    
    return model_path, history