"""
Prediction command for Voxelflex.

This module handles making predictions with trained RMSF models.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from voxelflex.data.data_loader import (
    load_voxel_data, 
    load_rmsf_data, 
    prepare_dataloaders,
    create_domain_mapping
)
from voxelflex.models.cnn_models import get_model
from voxelflex.utils.logging_utils import get_logger, ProgressBar
from voxelflex.utils.file_utils import ensure_dir
from voxelflex.utils.system_utils import get_device

logger = get_logger(__name__)

def predict_rmsf(
    config: Dict[str, Any],
    model_path: str,
    test_loader: Optional[DataLoader] = None
) -> str:
    """
    Make predictions with a trained model.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the trained model file
        test_loader: Optional test data loader. If None, one will be created.
        
    Returns:
        Path to the predictions file
    """
    # Get device (CPU or GPU)
    device = get_device(config["system_utilization"]["adjust_for_gpu"])
    logger.info(f"Using device: {device}")
    
    # Load the model
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model_config = checkpoint.get('config', {}).get('model', config['model'])
    input_shape = checkpoint.get('input_shape')
    
    if input_shape is None:
        logger.warning("Input shape not found in checkpoint, will try to infer from data")
    
    # Create and load the model
    model = get_model(
        architecture=model_config['architecture'],
        input_channels=model_config['input_channels'],
        channel_growth_rate=model_config['channel_growth_rate'],
        num_residual_blocks=model_config['num_residual_blocks'],
        dropout_rate=model_config['dropout_rate'],
        base_filters=model_config['base_filters']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    
    # Load data if test_loader is not provided
    voxel_data = None
    rmsf_data = None
    domain_mapping = None
    
    if test_loader is None:
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
        
        # Create domain mapping for later use
        voxel_domains = list(voxel_data.keys())
        rmsf_domains = rmsf_data['domain_id'].unique().tolist()
        domain_mapping = create_domain_mapping(voxel_domains, rmsf_domains)
        
        # Prepare dataloaders
        num_workers = config.get("system_utilization", {}).get("num_workers", 4)
        _, _, test_loader = prepare_dataloaders(
            voxel_data,
            rmsf_data,
            batch_size=config["training"]["batch_size"],
            train_split=config["training"]["train_split"],
            val_split=config["training"]["val_split"],
            test_split=config["training"]["test_split"],
            num_workers=num_workers,
            seed=config.get("training", {}).get("seed", 42)
        )
    else:
        # If test_loader is provided, we still need to load RMSF data for residue types
        rmsf_data = load_rmsf_data(
            config["input"]["rmsf_dir"],
            config["input"].get("replica", "replica_average"),
            config["input"]["temperature"]
        )
        
        # Try to access domain mapping from the dataset
        if hasattr(test_loader.dataset, 'domain_mapping'):
            domain_mapping = test_loader.dataset.domain_mapping
    
    # Make predictions
    all_predictions = []
    all_targets = []
    all_domain_ids = []
    all_resids = []
    
    show_progress = config["logging"]["show_progress_bars"]
    
    if show_progress:
        progress = ProgressBar(total=len(test_loader), prefix="Predicting", suffix="Complete")
    
    logger.info("Making predictions")
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            # Move data to device
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get domain and resid information
            if hasattr(test_loader.dataset, 'samples'):
                # Calculate the current batch's indices
                start_idx = i * test_loader.batch_size
                end_idx = min(start_idx + inputs.size(0), len(test_loader.dataset))
                batch_indices = list(range(start_idx, end_idx))
                
                # Get the domain_id and resid for each item in the current batch
                batch_domains = []
                batch_resids = []
                for idx in batch_indices:
                    domain_id, resid = test_loader.dataset.samples[idx]
                    batch_domains.append(domain_id)
                    batch_resids.append(resid)
                
                # Make sure we have the right number of items
                assert len(batch_domains) == outputs.size(0), f"Domain count mismatch: {len(batch_domains)} vs {outputs.size(0)}"
                
                # Add to our lists
                all_domain_ids.extend(batch_domains)
                all_resids.extend(batch_resids)
            
            # Store predictions and targets (using size(0) to ensure correct batch size handling)
            all_predictions.extend(outputs.cpu().numpy().flatten().tolist())
            all_targets.extend(targets.numpy().flatten().tolist())
            
            if show_progress:
                progress.update(i + 1)
    
    if show_progress:
        progress.finish()
    
    # Verify all lists have the same length
    logger.debug(f"Collected {len(all_predictions)} predictions")
    
    if all_domain_ids:
        logger.debug(f"Domains: {len(all_domain_ids)}, Resids: {len(all_resids)}, " 
                    f"Predictions: {len(all_predictions)}, Targets: {len(all_targets)}")
    
    # Create a DataFrame with predictions
    if all_domain_ids and len(all_domain_ids) == len(all_predictions):
        # If we have domain and residue information and lengths match
        predictions_df = pd.DataFrame({
            'domain_id': all_domain_ids,
            'resid': all_resids,
            'predicted_rmsf': all_predictions,
            'actual_rmsf': all_targets
        })
        
        # Convert resid to integer if it's a string
        if predictions_df['resid'].dtype == object:
            predictions_df['resid'] = predictions_df['resid'].astype(int)
        
        # Add residue names by joining with RMSF data
        if rmsf_data is not None:
            # Create a lookup DataFrame to join with
            residue_lookup = rmsf_data[['domain_id', 'resid', 'resname']].drop_duplicates()
            
            # Add a column with the base domain ID (strip _pdb, _pdb_clean, etc.)
            predictions_df['base_domain_id'] = predictions_df['domain_id'].apply(
                lambda x: x.split('_')[0] if '_' in x else x
            )
            
            # Try joining using base domain ID first
            merged_df = pd.merge(
                predictions_df,
                residue_lookup,
                left_on=['base_domain_id', 'resid'],
                right_on=['domain_id', 'resid'],
                how='left',
                suffixes=('', '_rmsf')
            )
            
            # Keep original domain_id and drop the extra columns
            if 'domain_id_rmsf' in merged_df.columns:
                merged_df = merged_df.drop('domain_id_rmsf', axis=1)
            
            # Drop the temporary base_domain_id column
            predictions_df = merged_df.drop('base_domain_id', axis=1)
            
            # Check if we have residue names
            if 'resname' in predictions_df.columns:
                resname_count = predictions_df['resname'].notna().sum()
                logger.info(f"Added residue names for {resname_count}/{len(predictions_df)} predictions")
    elif all_domain_ids:
        # If lengths don't match, use the minimum length
        min_length = min(len(all_domain_ids), len(all_resids), len(all_predictions), len(all_targets))
        logger.warning(f"Length mismatch in prediction arrays. Using minimum length: {min_length}")
        
        predictions_df = pd.DataFrame({
            'domain_id': all_domain_ids[:min_length],
            'resid': all_resids[:min_length],
            'predicted_rmsf': all_predictions[:min_length],
            'actual_rmsf': all_targets[:min_length]
        })
        
        # Add residue names using the same approach as above
        if rmsf_data is not None:
            residue_lookup = rmsf_data[['domain_id', 'resid', 'resname']].drop_duplicates()
            
            # Convert resid to integer if it's a string
            if predictions_df['resid'].dtype == object:
                predictions_df['resid'] = predictions_df['resid'].astype(int)
            
            # Add a column with the base domain ID
            predictions_df['base_domain_id'] = predictions_df['domain_id'].apply(
                lambda x: x.split('_')[0] if '_' in x else x
            )
            
            # Try joining using base domain ID
            merged_df = pd.merge(
                predictions_df,
                residue_lookup,
                left_on=['base_domain_id', 'resid'],
                right_on=['domain_id', 'resid'],
                how='left',
                suffixes=('', '_rmsf')
            )
            
            # Keep original domain_id and drop the extra columns
            if 'domain_id_rmsf' in merged_df.columns:
                merged_df = merged_df.drop('domain_id_rmsf', axis=1)
            
            # Drop the temporary base_domain_id column
            predictions_df = merged_df.drop('base_domain_id', axis=1)
            
            if 'resname' in predictions_df.columns:
                resname_count = predictions_df['resname'].notna().sum()
                logger.info(f"Added residue names for {resname_count}/{len(predictions_df)} predictions")
    else:
        # If we don't have domain and residue information
        predictions_df = pd.DataFrame({
            'sample_idx': list(range(len(all_predictions))),
            'predicted_rmsf': all_predictions,
            'actual_rmsf': all_targets
        })
    
    # Save predictions
    predictions_dir = os.path.join(config["output"]["base_dir"], "metrics")
    ensure_dir(predictions_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    predictions_path = os.path.join(predictions_dir, f"predictions_{timestamp}.csv")
    
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")
    
    return predictions_path