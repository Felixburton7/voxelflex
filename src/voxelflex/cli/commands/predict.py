"""
Prediction command for Voxelflex (Memory-Optimized).

This version uses domain streaming and chunked loading akin to 'train.py' to
prevent memory blowups for large test sets.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import h5py
import psutil
from torch.utils.data import DataLoader

from voxelflex.data.data_loader import (
    load_rmsf_data,
    RMSFDataset,
    create_domain_mapping,
    create_optimized_rmsf_lookup,
    check_memory_usage,
    clear_memory,
)
from voxelflex.models.cnn_models import get_model
from voxelflex.utils.logging_utils import (
    get_logger,
    EnhancedProgressBar,
    log_memory_usage,
)
from voxelflex.utils.file_utils import ensure_dir
from voxelflex.utils.system_utils import (
    get_device,
    set_num_threads,
    is_memory_critical,
    adjust_workers_for_memory,
    emergency_memory_reduction
)

# If you have a function like 'load_domain_batch' or 'prepare_dataloaders'
# in your codebase (from train.py), import and reuse it here:
from voxelflex.data.data_loader import load_domain_batch

from voxelflex.utils.system_utils import MEMORY_WARNING_THRESHOLD, MEMORY_CRITICAL_THRESHOLD, MEMORY_EMERGENCY_THRESHOLD


logger = get_logger(__name__)


def predict_rmsf(
    config: Dict[str, Any],
    model_path: str,
    domain_ids: Optional[List[str]] = None
) -> str:
    """
    Make predictions with a trained model using domain streaming with improved memory management.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the trained model file
        domain_ids: Optional subset of domain IDs to predict on.
                   If not provided, will infer from all domains in the voxel file.
        
    Returns:
        Path to the predictions CSV file
    """
    logger.info("=" * 60)
    logger.info("STARTING MEMORY-OPTIMIZED PREDICTION PROCESS")
    logger.info("=" * 60)
    logger.info(f"Model checkpoint: {model_path}")
    
    # 1. Check memory right away
    memory_stats = check_memory_usage()
    logger.info(f"Initial memory usage: System: {memory_stats['system_percent']}%, "
                f"Process: {memory_stats['process_rss_gb']:.2f} GB")
    
    # MEMORY OPTIMIZATION: If memory is already high, try to reduce it
    if memory_stats['system_percent'] > MEMORY_WARNING_THRESHOLD * 100:
        logger.warning(f"Starting prediction with high memory usage: {memory_stats['system_percent']}%")
        logger.warning("Attempting memory reduction before continuing")
        emergency_memory_reduction()
        
        # Check again after reduction
        memory_stats = check_memory_usage()
        if memory_stats['system_percent'] > MEMORY_CRITICAL_THRESHOLD * 100:
            logger.error(f"Cannot start prediction with memory usage at {memory_stats['system_percent']}%")
            logger.error("Please free up system memory before running prediction")
            raise MemoryError("Insufficient memory to begin prediction")

    # 2. Get device (CPU/GPU)
    device = get_device(config["system_utilization"]["adjust_for_gpu"])
    logger.info(f"Using device: {device}")
    
    # 3. Load the model checkpoint
    clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
    logger.info(f"Loading model from {model_path}")
    
    try:
        # MEMORY OPTIMIZATION: Load checkpoint carefully
        checkpoint = torch.load(model_path, map_location='cpu')  # Initially load to CPU
        model_config = checkpoint.get('config', {}).get('model', config['model'])
        input_shape = checkpoint.get('input_shape')
        
        # Create the model
        model = get_model(
            architecture=model_config['architecture'],
            input_channels=model_config['input_channels'],
            channel_growth_rate=model_config['channel_growth_rate'],
            num_residual_blocks=model_config['num_residual_blocks'],
            dropout_rate=model_config['dropout_rate'],
            base_filters=model_config['base_filters']
        )
        
        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Only now move to device
        model.to(device)
        model.eval()
        
        # MEMORY OPTIMIZATION: Extract only what we need, then delete checkpoint
        processed_domains = checkpoint.get('processed_domains', [])
        logger.info(f"Model was trained on {len(processed_domains)} domains")
        
        del checkpoint
        clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    # 4. Load RMSF data in a memory-efficient way
    logger.info("Loading RMSF data for residue information...")
    try:
        rmsf_data = load_rmsf_data(
            rmsf_dir=config["input"]["rmsf_dir"],
            replica=config["input"].get("replica", "replica_average"),
            temperature=config["input"]["temperature"]
        )
        # Create a global RMSF lookup to speed up merges
        global_rmsf_lookup = create_optimized_rmsf_lookup(rmsf_data)
        
        # MEMORY OPTIMIZATION: Check memory after loading RMSF data
        memory_stats = check_memory_usage()
        logger.info(f"Memory after loading RMSF data: {memory_stats['system_percent']}%")
        
        if memory_stats['system_percent'] > MEMORY_CRITICAL_THRESHOLD * 100:
            logger.warning(f"Critical memory usage after loading RMSF data: {memory_stats['system_percent']}%")
            emergency_memory_reduction()
    except Exception as e:
        logger.error(f"Error loading RMSF data: {str(e)}")
        raise
    
    # 5. Figure out which domains we are predicting. If domain_ids not given, use all.
    voxel_file = config["input"]["voxel_file"]
    if not os.path.exists(voxel_file):
        raise FileNotFoundError(f"Voxel file not found: {voxel_file}")
    
    try:
        all_available_domains = []
        with h5py.File(voxel_file, 'r') as f:
            all_available_domains = list(f.keys())
        
        logger.info(f"Found {len(all_available_domains)} domains in HDF5 file")
        
        if domain_ids and len(domain_ids) > 0:
            # Filter the HDF5 domains based on the user-provided domain list
            all_domains = []
            for d in all_available_domains:
                base_d = d.split('_')[0]
                if base_d in domain_ids or d in domain_ids:
                    all_domains.append(d)
            if not all_domains:
                logger.warning("None of the specified domain_ids found. Using all available domains instead.")
                all_domains = all_available_domains
        else:
            # Use everything
            all_domains = all_available_domains
        
        logger.info(f"Total domains available for prediction: {len(all_domains)}")
    except Exception as e:
        logger.error(f"Error reading domains from HDF5 file: {str(e)}")
        raise
    
    # 6. MEMORY OPTIMIZATION: Decide how many domains to load per batch based on memory
    memory_stats = check_memory_usage()
    system_memory_percent = memory_stats["system_percent"]
    
    # Calculate domains_per_batch conservatively based on current memory pressure
    if system_memory_percent > 70:  # High memory pressure
        domains_per_batch = min(10, config["prediction"].get("domains_per_batch", 50) // 4)
        logger.warning(f"High memory pressure ({system_memory_percent:.1f}%), using reduced batch size of {domains_per_batch} domains")
    elif system_memory_percent > 60:  # Moderate memory pressure
        domains_per_batch = min(20, config["prediction"].get("domains_per_batch", 50) // 2)
        logger.warning(f"Moderate memory pressure ({system_memory_percent:.1f}%), using reduced batch size of {domains_per_batch} domains")
    else:  # Normal memory pressure
        domains_per_batch = min(50, config["prediction"].get("domains_per_batch", 50))
        logger.info(f"Normal memory usage ({system_memory_percent:.1f}%), using batch size of {domains_per_batch} domains")
    
    logger.info(f"Using domain batch size: {domains_per_batch}")
    
    # 7. Create domain batches
    domain_indices = np.arange(len(all_domains))
    test_domain_batches = [
        domain_indices[i : i + domains_per_batch]
        for i in range(0, len(domain_indices), domains_per_batch)
    ]
    logger.info(f"Created {len(test_domain_batches)} domain batches for prediction")
    
    # 8. MEMORY OPTIMIZATION: Prepare to store predictions more efficiently
    # Instead of accumulating in memory, write directly to disk in chunks
    predictions_dir = os.path.join(config["output"]["base_dir"], "metrics")
    ensure_dir(predictions_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    predictions_path = os.path.join(predictions_dir, f"predictions_{timestamp}.csv")
    
    # Write CSV header
    with open(predictions_path, 'w') as f:
        f.write("domain_id,resid,resname,predicted_rmsf,actual_rmsf\n")
    
    # 9. For progress logging
    show_progress = config["logging"]["show_progress_bars"]
    outer_progress = None
    if show_progress:
        outer_progress = EnhancedProgressBar(
            total=len(test_domain_batches),
            prefix="Overall Prediction Batches",
            suffix="Complete",
            stage_info="PREDICTION"
        )
    
    # 10. MEMORY OPTIMIZATION: Track batches processed and total predictions for reporting
    batches_processed = 0
    total_predictions = 0
    
    # 11. Predict in a loop over domain batches
    with torch.no_grad():
        for batch_idx, domain_batch_idx in enumerate(test_domain_batches):
            batch_start_time = time.time()
            
            # MEMORY OPTIMIZATION: Check memory before loading batch
            memory_stats = check_memory_usage()
            if memory_stats['system_percent'] > MEMORY_CRITICAL_THRESHOLD * 100:
                logger.warning(f"Critical memory usage ({memory_stats['system_percent']:.1f}%) before batch {batch_idx+1}")
                logger.warning("Performing emergency memory reduction")
                emergency_memory_reduction()
                
                # If still critical, we need to skip this batch
                memory_stats = check_memory_usage()
                if memory_stats['system_percent'] > MEMORY_EMERGENCY_THRESHOLD * 100:
                    logger.error(f"Memory usage still critical ({memory_stats['system_percent']:.1f}%) after reduction")
                    logger.error(f"Skipping batch {batch_idx+1} to prevent system crash")
                    
                    if outer_progress:
                        outer_progress.update(batch_idx+1)
                    
                    continue
            
            batch_domains = [all_domains[i] for i in domain_batch_idx]
            
            logger.info(f"\n--- Processing Batch {batch_idx+1}/{len(test_domain_batches)} with {len(batch_domains)} domains ---")
            
            # (a) Load domain batch from HDF5
            try:
                domain_voxel_data = load_domain_batch(domain_batch_idx, all_domains, config)
                
                if not domain_voxel_data:
                    logger.warning("No voxel data loaded for this batch. Skipping.")
                    if outer_progress:
                        outer_progress.update(batch_idx+1)
                    continue
            except Exception as e:
                logger.error(f"Error loading domain batch: {str(e)}")
                if outer_progress:
                    outer_progress.update(batch_idx+1)
                continue
            
            # Check memory after loading domains
            memory_stats = check_memory_usage()
            logger.info(f"Memory after loading domains: {memory_stats['system_percent']}%")
            
            # (b) Create domain mapping
            voxel_domains = list(domain_voxel_data.keys())
            rmsf_domains = rmsf_data['domain_id'].unique().tolist()
            domain_mapping = create_domain_mapping(voxel_domains, rmsf_domains)
            
            # (c) Build a memory-optimized dataset
            try:
                dataset = RMSFDataset(
                    voxel_data=domain_voxel_data,
                    rmsf_data=rmsf_data,
                    domain_mapping=domain_mapping,
                    transform=None,
                    memory_efficient=True,  # CHANGE: Always use memory-efficient mode
                    global_rmsf_lookup=global_rmsf_lookup
                )
            except Exception as e:
                logger.error(f"Error creating dataset: {str(e)}")
                del domain_voxel_data
                clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
                if outer_progress:
                    outer_progress.update(batch_idx+1)
                continue
            
            # (d) MEMORY OPTIMIZATION: Use much smaller batch size for DataLoader
            memory_stats = check_memory_usage()
            if memory_stats['system_percent'] > 80:
                # Very high memory pressure - use minimal batch size
                predicted_batch_size = 8
                logger.warning(f"Very high memory pressure ({memory_stats['system_percent']:.1f}%). Using minimal batch size of {predicted_batch_size}.")
            elif memory_stats['system_percent'] > 70:
                # High memory pressure - use small batch size
                predicted_batch_size = 16
                logger.warning(f"High memory pressure ({memory_stats['system_percent']:.1f}%). Using reduced batch size of {predicted_batch_size}.")
            else:
                # Normal memory pressure - use moderate batch size
                predicted_batch_size = min(32, config["prediction"].get("batch_size", 32))
                logger.info(f"Using batch size: {predicted_batch_size}")
            
            # (e) MEMORY OPTIMIZATION: Use minimal workers
            num_workers = config['system_utilization']['num_workers']
            
            logger.info(f"DataLoader batch_size={predicted_batch_size}, num_workers={num_workers}")
            
            try:
                test_loader = DataLoader(
                    dataset,
                    batch_size=predicted_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=(device.type == 'cuda'),
                    persistent_workers=False,
                )
            except Exception as e:
                logger.error(f"Error creating DataLoader: {str(e)}")
                del domain_voxel_data, dataset
                clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
                if outer_progress:
                    outer_progress.update(batch_idx+1)
                continue
            
            # (f) MEMORY OPTIMIZATION: Predict and write directly to file in chunks
            batch_predictions = []
            batch_progress = None
            if show_progress:
                batch_progress = EnhancedProgressBar(
                    total=len(test_loader),
                    prefix=f"Predicting Batch {batch_idx+1}",
                    suffix="Complete",
                    stage_info="BATCH_PRED"
                )
            
            # For chunked file writing
            chunk_size = 1000
            predictions_chunk = []
            total_batch_predictions = 0
            
            sample_offset = 0
            try:
                for i, (inputs, targets) in enumerate(test_loader):
                    # MEMORY OPTIMIZATION: Check memory inside prediction loop
                    if i > 0 and i % 20 == 0:
                        memory_stats = check_memory_usage()
                        if memory_stats['system_percent'] > MEMORY_CRITICAL_THRESHOLD * 100:
                            logger.warning(f"Critical memory usage ({memory_stats['system_percent']:.1f}%) during prediction. Emergency reduction...")
                            emergency_memory_reduction()
                    
                    inputs = inputs.to(device)
                    
                    # forward pass
                    outputs = model(inputs)
                    
                    # Move predictions and targets back to CPU
                    outputs_cpu = outputs.cpu().numpy()
                    targets_cpu = targets.cpu().numpy()
                    
                    # Domain/residue info from dataset
                    start_idx = sample_offset
                    end_idx = sample_offset + len(inputs)
                    sample_indices = range(start_idx, end_idx)
                    
                    # Gather per-sample predictions as CSV rows
                    for idx_in_batch, sample_idx_in_dataset in enumerate(sample_indices):
                        try:
                            domain_id, resid = dataset.samples[sample_idx_in_dataset]
                            pred_val = float(outputs_cpu[idx_in_batch])
                            true_val = float(targets_cpu[idx_in_batch])
                            
                            # Try to get residue name if available
                            resname = "UNK"
                            try:
                                # Check if resname is available via lookup
                                lookup_key = (domain_mapping.get(domain_id, domain_id), int(resid))
                                if lookup_key in global_rmsf_lookup:
                                    domain_filter = rmsf_data['domain_id'] == lookup_key[0]
                                    resid_filter = rmsf_data['resid'] == lookup_key[1]
                                    matching_rows = rmsf_data[domain_filter & resid_filter]
                                    if not matching_rows.empty:
                                        resname = matching_rows.iloc[0]['resname']
                            except Exception:
                                pass  # Keep default "UNK" if lookup fails
                            
                            # Add to chunk
                            predictions_chunk.append(f"{domain_id},{resid},{resname},{pred_val:.6f},{true_val:.6f}")
                            total_batch_predictions += 1
                            
                            # Write chunk to file when it reaches chunk_size
                            if len(predictions_chunk) >= chunk_size:
                                with open(predictions_path, 'a') as f:
                                    f.write('\n'.join(predictions_chunk) + '\n')
                                predictions_chunk = []
                        except Exception as e:
                            logger.warning(f"Error processing prediction for sample {sample_idx_in_dataset}: {str(e)}")
                    
                    sample_offset += len(inputs)
                    
                    # Clean up
                    del inputs, outputs, outputs_cpu, targets_cpu
                    
                    if batch_progress:
                        batch_progress.update(i + 1)
                    
                    # MEMORY OPTIMIZATION: Periodically clear CUDA cache
                    if (i + 1) % 5 == 0 and device.type == 'cuda':
                        torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
            
            # Write any remaining predictions
            if predictions_chunk:
                with open(predictions_path, 'a') as f:
                    f.write('\n'.join(predictions_chunk) + '\n')
            
            if batch_progress:
                batch_progress.finish()
            
            # Update total predictions count
            total_predictions += total_batch_predictions
            logger.info(f"Processed {total_batch_predictions} predictions from batch {batch_idx+1}")
            
            # Done with this domain batch. Clean up, free memory.
            del domain_voxel_data, dataset, test_loader
            clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
            
            batches_processed += 1
            batch_duration = time.time() - batch_start_time
            logger.info(f"Finished Batch {batch_idx+1}/{len(test_domain_batches)} in {batch_duration:.2f}s")
            
            if outer_progress:
                outer_progress.update(batch_idx+1)
                
            
            # MEMORY OPTIMIZATION: Check if system memory is critical, exit gracefully if needed
            memory_stats = check_memory_usage()
            if memory_stats["system_percent"] > MEMORY_EMERGENCY_THRESHOLD * 100:
                logger.critical(f"System memory critically high ({memory_stats['system_percent']:.1f}%), stopping predictions to prevent crash")
                logger.critical(f"Processed {batches_processed}/{len(test_domain_batches)} batches before stopping")
                break
    
    if outer_progress:
        outer_progress.finish()
    
    logger.info(f"\nCompleted prediction with {total_predictions} total predictions from {batches_processed} batches")
    logger.info(f"Predictions saved to {predictions_path}")
    logger.info("PREDICTION PROCESS COMPLETED SUCCESSFULLY")
    
    # Final memory cleanup
    clear_memory(force_gc=True, clear_cuda=True, aggressive=True)
    
    return predictions_path