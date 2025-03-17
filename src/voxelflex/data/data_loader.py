"""
Data loading module for Voxelflex.

This module handles loading and processing of voxelized protein data (.hdf5)
and RMSF data (.csv).
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from voxelflex.data.validators import (
    validate_voxel_data,
    validate_rmsf_data,
    validate_domain_residue_mapping
)
from voxelflex.utils.logging_utils import get_logger
from voxelflex.utils.file_utils import resolve_path

logger = get_logger(__name__)

class RMSFDataset(Dataset):
    """PyTorch Dataset for voxel and RMSF data."""
    
    def __init__(
        self,
        voxel_data: Dict[str, Dict[str, np.ndarray]],
        rmsf_data: pd.DataFrame,
        domain_mapping: Dict[str, str],
        transform=None
    ):
        """
        Initialize RMSF dataset.
        
        Args:
            voxel_data: Dictionary mapping domain_ids to voxel data
            rmsf_data: DataFrame containing RMSF values
            domain_mapping: Mapping from hdf5 domain to RMSF domain
            transform: Optional transforms to apply
        """
        self.voxel_data = voxel_data
        self.rmsf_data = rmsf_data
        self.domain_mapping = domain_mapping
        self.transform = transform
        
        # Create a list of (domain_id, residue_id) tuples for indexing
        self.samples = []
        for domain_id, domain_data in voxel_data.items():
            for resid in domain_data:
                try:
                    rmsf_domain = self.domain_mapping.get(domain_id, domain_id)
                    # Check if this (domain, residue) pair exists in RMSF data
                    rmsf_value = self.rmsf_data[
                        (self.rmsf_data['domain_id'] == rmsf_domain) & 
                        (self.rmsf_data['resid'] == int(resid))
                    ]['average_rmsf']
                    
                    if len(rmsf_value) > 0:
                        self.samples.append((domain_id, resid))
                except Exception as e:
                    logger.debug(f"Skipping {domain_id}:{resid} - {str(e)}")
        
        logger.info(f"Created dataset with {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        domain_id, resid = self.samples[idx]
        voxel = self.voxel_data[domain_id][resid]
        
        # Get the corresponding RMSF value
        rmsf_domain = self.domain_mapping.get(domain_id, domain_id)
        rmsf_value = self.rmsf_data[
            (self.rmsf_data['domain_id'] == rmsf_domain) & 
            (self.rmsf_data['resid'] == int(resid))
        ]['average_rmsf'].values[0]
        
        # Convert to tensors
        voxel_tensor = torch.tensor(voxel, dtype=torch.float32)
        rmsf_tensor = torch.tensor(rmsf_value, dtype=torch.float32)
        
        if self.transform:
            voxel_tensor = self.transform(voxel_tensor)
            
        return voxel_tensor, rmsf_tensor


def load_voxel_data(
    voxel_file: str,
    domain_ids: Optional[List[str]] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load voxel data from HDF5 file.
    
    Args:
        voxel_file: Path to HDF5 file
        domain_ids: Optional list of domain IDs to load. If None, load all domains.
        
    Returns:
        Dictionary mapping domain IDs to dictionaries of residue voxel data
    """
    voxel_file = resolve_path(voxel_file)
    logger.info(f"Loading voxel data from {voxel_file}")
    
    if not os.path.exists(voxel_file):
        raise FileNotFoundError(f"Voxel file not found: {voxel_file}")
    
    voxel_data = {}
    
    with h5py.File(voxel_file, 'r') as f:
        # Get list of all domains in the file
        available_domains = list(f.keys())
        logger.info(f"Found {len(available_domains)} domains in voxel file")
        
        # Filter domains if domain_ids is provided and not empty
        if domain_ids and len(domain_ids) > 0:
            # Extract base domain_id from available domains for matching
            domains_to_process = []
            for domain in available_domains:
                # Extract base domain (remove _pdb, _pdb_clean, etc.)
                base_domain = domain.split('_')[0]
                if base_domain in domain_ids:
                    domains_to_process.append(domain)
            
            if not domains_to_process:
                logger.warning(f"None of the specified domain_ids were found in the voxel file")
                # Show some examples to help with debugging
                sample_domains = [d.split('_')[0] for d in available_domains[:10]]
                logger.debug(f"First few available domain IDs (base names): {sample_domains}...")
                logger.debug(f"Specified domain_ids: {domain_ids}")
        else:
            domains_to_process = available_domains
        
        logger.info(f"Processing {len(domains_to_process)} domains")
        
        # Process each domain
        for domain_id in domains_to_process:
            domain_group = f[domain_id]
            
            # Based on inspection, there's only one key per domain
            if len(domain_group.keys()) == 0:
                logger.warning(f"Domain {domain_id} has no children")
                continue
                
            # Get the first direct child (this could be '0', 'A', etc.)
            first_child_key = list(domain_group.keys())[0]
            residue_group = domain_group[first_child_key]
            
            logger.debug(f"Using residue group '{first_child_key}' for domain {domain_id}")
            
            # Load residue voxel data
            domain_data = {}
            
            # Get all residue IDs (numeric keys)
            residue_keys = [k for k in residue_group.keys() if isinstance(k, str) and k.isdigit()]
            logger.debug(f"Found {len(residue_keys)} residues for domain {domain_id}")
            
            for resid in residue_keys:
                try:
                    # Based on the inspection, residue_data is a direct dataset
                    residue_data = residue_group[resid]
                    
                    if isinstance(residue_data, h5py.Dataset):
                        # Get the dataset shape
                        shape = residue_data.shape
                        
                        # Load the data
                        voxel_data_raw = residue_data[:]
                        
                        # For datasets with shape (x, y, z, channels), transpose to (channels, x, y, z)
                        if len(shape) == 4 and shape[3] in [4, 5]:  # Last dimension is channels
                            voxel = np.transpose(voxel_data_raw, (3, 0, 1, 2))
                            logger.debug(f"Transposed voxel shape from {shape} to {voxel.shape} for residue {resid}")
                        else:
                            # If not the expected 4D shape, just use as-is and log
                            voxel = voxel_data_raw
                            logger.debug(f"Using original shape {shape} for residue {resid}")
                        
                        # Convert to float32 for compatibility with PyTorch
                        if voxel.dtype == bool:
                            voxel = voxel.astype(np.float32)
                        
                        domain_data[resid] = voxel
                    else:
                        logger.debug(f"Skipping residue {resid}: not a dataset")
                        
                except Exception as e:
                    logger.warning(f"Error processing residue {resid} for domain {domain_id}: {str(e)}")
            
            if domain_data:
                # Store with the original domain ID to maintain traceability
                voxel_data[domain_id] = domain_data
                logger.info(f"Loaded {len(domain_data)} residues for domain {domain_id}")
            else:
                logger.warning(f"No voxel data found for domain {domain_id}")
    
    if not voxel_data:
        logger.error("No valid voxel data was loaded from the file")
        # Provide more diagnostic information
        logger.error(f"Attempted to process: {domains_to_process}")
        raise ValueError("No valid voxel data was loaded. Check the logs for details.")
    
    # Validate the loaded voxel data
    voxel_data = validate_voxel_data(voxel_data)
    
    return voxel_data

def load_rmsf_data(
    rmsf_dir: str,
    replica: str = "replica_average",
    temperature: Union[int, str] = 320
) -> pd.DataFrame:
    """
    Load RMSF data from CSV file.
    
    Args:
        rmsf_dir: Base directory for RMSF data
        replica: Replica folder name (default: "replica_average")
        temperature: Temperature value (default: 320)
        
    Returns:
        DataFrame containing RMSF data
    """
    rmsf_dir = resolve_path(rmsf_dir)
    logger.info(f"Loading RMSF data from {rmsf_dir}, replica={replica}, temperature={temperature}")
    
    # Construct the path to the RMSF CSV file
    if isinstance(temperature, int):
        temperature_str = str(temperature)
    else:
        temperature_str = temperature
    
    rmsf_file = os.path.join(
        rmsf_dir, 
        replica, 
        temperature_str,
        f"rmsf_{replica}_temperature{temperature_str}.csv"
    )
    
    if not os.path.exists(rmsf_file):
        raise FileNotFoundError(f"RMSF file not found: {rmsf_file}")
    
    # Load the CSV file
    rmsf_data = pd.read_csv(rmsf_file)
    
    # Validate the RMSF data
    rmsf_data = validate_rmsf_data(rmsf_data)
    
    logger.info(f"Loaded RMSF data with {len(rmsf_data)} entries")
    return rmsf_data


def create_domain_mapping(
    voxel_domains: List[str],
    rmsf_domains: List[str]
) -> Dict[str, str]:
    """
    Create mapping between voxel domain IDs and RMSF domain IDs.
    
    Args:
        voxel_domains: List of domain IDs from voxel data
        rmsf_domains: List of domain IDs from RMSF data
        
    Returns:
        Dictionary mapping voxel domain IDs to RMSF domain IDs
    """
    logger.info("Creating domain mapping")
    
    mapping = {}
    
    # First, try direct matches
    for voxel_domain in voxel_domains:
        if voxel_domain in rmsf_domains:
            mapping[voxel_domain] = voxel_domain
        else:
            # Try to extract a "clean" domain name (remove _pdb, _pdb_clean, etc.)
            clean_domain = voxel_domain.split('_')[0]
            if clean_domain in rmsf_domains:
                mapping[voxel_domain] = clean_domain
            else:
                logger.warning(f"No matching RMSF domain found for voxel domain {voxel_domain} (base name: {clean_domain})")
    
    # Report mapping statistics
    total_mapped = len(mapping)
    mapping_percentage = (total_mapped / len(voxel_domains)) * 100 if voxel_domains else 0
    
    logger.info(f"Created mapping for {total_mapped} domains out of {len(voxel_domains)} "
                f"({mapping_percentage:.1f}%)")
    
    # If no mappings were created, provide more detailed information
    if not mapping:
        logger.error("No domain mappings could be created!")
        logger.error("This could be due to naming inconsistencies between voxel and RMSF data")
        logger.error(f"Sample voxel domains: {voxel_domains[:5]}")
        logger.error(f"Sample RMSF domains: {rmsf_domains[:5]}")
        logger.error("Ensure that either the full domain names match or the base names match")
        
        # Try more permissive matching as a fallback
        logger.info("Attempting more permissive matching as fallback...")
        for voxel_domain in voxel_domains:
            clean_voxel = voxel_domain.split('_')[0]
            # Try to find any RMSF domain that starts with the same characters
            for rmsf_domain in rmsf_domains:
                if rmsf_domain.startswith(clean_voxel) or clean_voxel.startswith(rmsf_domain):
                    mapping[voxel_domain] = rmsf_domain
                    logger.debug(f"Fallback mapping: {voxel_domain} -> {rmsf_domain}")
                    break
        
        if mapping:
            logger.info(f"Created {len(mapping)} fallback mappings")
    
    return mapping


def prepare_dataloaders(
    voxel_data: Dict[str, Dict[str, np.ndarray]],
    rmsf_data: pd.DataFrame,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        voxel_data: Dictionary mapping domain IDs to voxel data
        rmsf_data: DataFrame containing RMSF values
        batch_size: Batch size for DataLoaders
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        num_workers: Number of worker processes for DataLoaders
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("Preparing DataLoaders")
    
    # Create domain mapping
    voxel_domains = list(voxel_data.keys())
    rmsf_domains = rmsf_data['domain_id'].unique().tolist()
    domain_mapping = create_domain_mapping(voxel_domains, rmsf_domains)
    
    # Validate the mapping
    validate_domain_residue_mapping(voxel_data, rmsf_data, domain_mapping)
    
    # Create dataset
    dataset = RMSFDataset(voxel_data, rmsf_data, domain_mapping)
    
    # Split the dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_end = int(train_split * dataset_size)
    val_end = train_end + int(val_split * dataset_size)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    logger.info(f"Split dataset into {len(train_indices)} training, "
                f"{len(val_indices)} validation, and {len(test_indices)} test samples")
    
    # Create DataLoaders
    train_loader = DataLoader(
        dataset, batch_size=batch_size, 
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_indices),
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader