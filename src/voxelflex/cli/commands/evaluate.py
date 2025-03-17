"""
Evaluation command for Voxelflex.

This module handles evaluating the performance of trained RMSF models.
"""

import os
import time
import json
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from voxelflex.utils.logging_utils import get_logger
from voxelflex.utils.file_utils import ensure_dir, save_json

logger = get_logger(__name__)

def evaluate_model(
    config: Dict[str, Any],
    model_path: str,
    predictions_path: Optional[str] = None
) -> str:
    """
    Evaluate model performance using various metrics.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the trained model file
        predictions_path: Path to predictions file. If None, predictions will be made.
        
    Returns:
        Path to the metrics file
    """
    logger.info("Evaluating model performance")
    
    # If predictions_path is not provided, generate predictions
    if predictions_path is None:
        from voxelflex.cli.commands.predict import predict_rmsf
        predictions_path = predict_rmsf(config, model_path)
    
    # Load predictions
    logger.info(f"Loading predictions from {predictions_path}")
    predictions_df = pd.read_csv(predictions_path)
    
    # Extract predictions and actual values
    y_true = predictions_df['actual_rmsf'].values
    y_pred = predictions_df['predicted_rmsf'].values
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate domain-level metrics if domain info is available
    domain_metrics = {}
    if 'domain_id' in predictions_df.columns:
        for domain in predictions_df['domain_id'].unique():
            domain_df = predictions_df[predictions_df['domain_id'] == domain]
            domain_y_true = domain_df['actual_rmsf'].values
            domain_y_pred = domain_df['predicted_rmsf'].values
            
            # Skip if we don't have enough data
            if len(domain_y_true) < 2:
                logger.warning(f"Not enough data points for domain {domain}, skipping metrics")
                continue
                
            domain_metrics[domain] = {
                'mse': mean_squared_error(domain_y_true, domain_y_pred),
                'rmse': np.sqrt(mean_squared_error(domain_y_true, domain_y_pred)),
                'mae': mean_absolute_error(domain_y_true, domain_y_pred),
                'r2': r2_score(domain_y_true, domain_y_pred),
                'num_residues': len(domain_df)
            }
    
    # Calculate residue type metrics if residue type info is available
    residue_type_metrics = {}
    if 'resname' in predictions_df.columns:
        # Get unique residue types with at least 2 data points
        valid_restypes = []
        for resname in predictions_df['resname'].dropna().unique():
            resname_df = predictions_df[predictions_df['resname'] == resname]
            if len(resname_df) >= 2:  # Need at least 2 points for correlation/R²
                valid_restypes.append(resname)
            else:
                logger.debug(f"Not enough data points for residue type {resname}, skipping metrics")
        
        # Calculate metrics for each valid residue type
        for resname in valid_restypes:
            resname_df = predictions_df[predictions_df['resname'] == resname]
            resname_y_true = resname_df['actual_rmsf'].values
            resname_y_pred = resname_df['predicted_rmsf'].values
            
            try:
                residue_type_metrics[resname] = {
                    'mse': mean_squared_error(resname_y_true, resname_y_pred),
                    'rmse': np.sqrt(mean_squared_error(resname_y_true, resname_y_pred)),
                    'mae': mean_absolute_error(resname_y_true, resname_y_pred),
                    'r2': r2_score(resname_y_true, resname_y_pred),
                    'num_residues': len(resname_df)
                }
            except Exception as e:
                logger.warning(f"Error calculating metrics for residue type {resname}: {str(e)}")
    
    # Log overall metrics
    logger.info(f"Overall Mean Squared Error (MSE): {mse:.6f}")
    logger.info(f"Overall Root Mean Squared Error (RMSE): {rmse:.6f}")
    logger.info(f"Overall Mean Absolute Error (MAE): {mae:.6f}")
    logger.info(f"Overall R²: {r2:.6f}")
    
    # Save metrics
    metrics = {
        'overall': {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'num_samples': len(y_true)
        },
        'domains': domain_metrics,
        'residue_types': residue_type_metrics
    }
    
    metrics_dir = os.path.join(config["output"]["base_dir"], "metrics")
    ensure_dir(metrics_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(metrics_dir, f"metrics_{timestamp}.json")
    
    save_json(metrics, metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")
    
    return metrics_path