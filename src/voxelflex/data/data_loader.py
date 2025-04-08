# """
# Data loading module for Voxelflex.

# This module handles loading and processing of voxelized protein data (.hdf5)
# and RMSF data (.csv).
# """

# import os
# import logging
# import h5py  # Make sure this import is here
# import psutil  # For memory monitoring
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional, Union, Any
# import time

# import h5py
# import numpy as np
# import pandas as pd
# import torch
# import multiprocessing
# from torch.utils.data import Dataset, DataLoader

# from voxelflex.data.validators import (
#     validate_voxel_data,
#     validate_rmsf_data,
#     validate_domain_residue_mapping
# )
# from voxelflex.utils.logging_utils import (
#     get_logger, 
#     setup_logging, 
#     EnhancedProgressBar, 
#     log_memory_usage, 
#     log_operation_result, 
#     log_section_header, 
#     log_stage, 
#     log_step
# )
# from voxelflex.utils.file_utils import resolve_path
# from voxelflex.utils.system_utils import (
#     get_device, 
#     check_system_resources, 
#     clear_memory, 
#     check_memory_usage,
#     set_num_threads, 
#     is_memory_critical, 
#     estimate_batch_size, 
#     adjust_workers_for_memory
# )

# logger = get_logger(__name__)

# class RMSFDataset(Dataset):
#     """PyTorch Dataset for voxel and RMSF data with dramatically improved performance."""
    
#     def __init__(
#         self,
#         voxel_data: Dict[str, Dict[str, np.ndarray]],
#         rmsf_data: pd.DataFrame,
#         domain_mapping: Dict[str, str],
#         transform=None,
#         memory_efficient: bool = False,  # Default to False for better performance
#         global_rmsf_lookup: Dict[Tuple[str, int], float] = None
#     ):
#         """
#         Initialize RMSF dataset with highly optimized performance.
        
#         Args:
#             voxel_data: Dictionary mapping domain_ids to voxel data
#             rmsf_data: DataFrame containing RMSF values
#             domain_mapping: Mapping from hdf5 domain to RMSF domain
#             transform: Optional transforms to apply
#             memory_efficient: If True, use memory-efficient mode (default: False)
#             global_rmsf_lookup: Pre-computed global RMSF lookup dictionary
#         """
#         self.voxel_data = voxel_data
#         self.domain_mapping = domain_mapping
#         self.transform = transform
#         self.memory_efficient = memory_efficient
        
#         # Keep original DataFrame for __getitem__ fallbacks
#         self.rmsf_data = rmsf_data
        
#         # Track timing for performance analysis
#         start_time = time.time()
        
#         # Use pre-computed lookup if provided
#         if global_rmsf_lookup is not None:
#             logger.info("Using pre-computed global RMSF lookup")
#             self.rmsf_lookup = global_rmsf_lookup
#             lookup_time = 0.0
#         else:
#             # Create RMSF lookup dictionary
#             logger.info("Creating optimized RMSF lookup dictionary")
            
#             # Extract columns as numpy arrays for faster processing
#             domains = rmsf_data['domain_id'].values
#             resids = rmsf_data['resid'].values
#             rmsfs = rmsf_data['average_rmsf'].values
            
#             # Create lookup in a single, efficient pass
#             self.rmsf_lookup = {}
#             for i in range(len(domains)):
#                 try:
#                     resid_int = int(resids[i])
#                     self.rmsf_lookup[(domains[i], resid_int)] = rmsfs[i]
#                 except (ValueError, TypeError):
#                     continue
            
#             lookup_time = time.time() - start_time
#             logger.info(f"Created RMSF lookup with {len(self.rmsf_lookup)} entries in {lookup_time:.2f} seconds")
            
#             # Create additional lookup dictionaries for faster access
#             base_domain_time = time.time()
            
#             # Create base domain lookup
#             self.base_domain_lookup = {}
#             for domain in set(domains):
#                 base_domain = domain.split('_')[0] if '_' in domain else domain
#                 self.base_domain_lookup[domain] = base_domain
            
#             # Create optimized base domain lookup for faster residue matching
#             self.base_rmsf_lookup = {}
#             for (domain, resid), rmsf in self.rmsf_lookup.items():
#                 base_domain = domain.split('_')[0] if '_' in domain else domain
#                 if base_domain not in self.base_rmsf_lookup:
#                     self.base_rmsf_lookup[base_domain] = {}
#                 self.base_rmsf_lookup[base_domain][resid] = rmsf
                
#             logger.debug(f"Created base domain lookups in {time.time() - base_domain_time:.2f} seconds")
        
#         # Create samples list with optimized algorithm
#         samples_start_time = time.time()
#         logger.info("Creating dataset samples list with optimized algorithm")
        
#         # Count total residues for pre-allocation (avoid resizing)
#         total_residues = sum(len(domain_data) for domain_data in voxel_data.values())
        
#         # Pre-allocate samples list with enough capacity
#         # Reserve approximately 80% of total residues (high estimated match rate)
#         self.samples = [None] * (total_residues * 8 // 10)
#         self.sample_count = 0
        
#         # Track statistics
#         total_domains = len(voxel_data)
#         matched_residues = 0
#         direct_matches = 0
#         base_matches = 0
        
#         # Process all domains and residues
#         for domain_id, domain_data in voxel_data.items():
#             mapped_domain = domain_mapping.get(domain_id, domain_id)
#             domain_matches = 0
            
#             # Process all residues in the domain
#             for resid in domain_data:
#                 try:
#                     resid_int = int(resid)
#                     lookup_key = (mapped_domain, resid_int)
#                     added = False
                    
#                     # First try direct lookup
#                     if lookup_key in self.rmsf_lookup:
#                         self._add_sample(domain_id, resid)
#                         domain_matches += 1
#                         matched_residues += 1
#                         direct_matches += 1
#                         added = True
                    
#                     # If not found directly, try with base domain name
#                     if not added:
#                         base_domain = domain_id.split('_')[0] if '_' in domain_id else domain_id
#                         alt_key = (base_domain, resid_int)
                        
#                         if alt_key in self.rmsf_lookup:
#                             self._add_sample(domain_id, resid)
#                             domain_matches += 1
#                             matched_residues += 1
#                             base_matches += 1
#                             added = True
#                 except Exception as e:
#                     # Skip invalid residues silently
#                     pass
        
#         # Resize samples list to actual count
#         self.samples = self.samples[:self.sample_count]
        
#         samples_time = time.time() - samples_start_time
#         total_time = time.time() - start_time
        
#         logger.info(f"Created dataset with {len(self.samples)} samples from {matched_residues} "
#                     f"residues across {total_domains} domains in {total_time:.2f} seconds")
#         logger.info(f"Direct matches: {direct_matches}, Base name matches: {base_matches}")
    
#     def _add_sample(self, domain_id, resid):
#         """Helper method to add a sample to the samples list."""
#         if self.sample_count < len(self.samples):
#             self.samples[self.sample_count] = (domain_id, resid)
#         else:
#             self.samples.append((domain_id, resid))
#         self.sample_count += 1
    
#     def __len__(self) -> int:
#         return len(self.samples)
    
#     def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Get a data sample with improved error handling and lookup performance."""
#         domain_id, resid = self.samples[idx]
        
#         try:
#             # Access voxel data
#             voxel = self.voxel_data[domain_id][resid]
            
#             # Get the corresponding RMSF value using optimized lookup
#             rmsf_domain = self.domain_mapping.get(domain_id, domain_id)
#             lookup_key = (rmsf_domain, int(resid))
            
#             if lookup_key in self.rmsf_lookup:
#                 rmsf_value = self.rmsf_lookup[lookup_key]
#             else:
#                 # Try alternative lookup strategies
#                 alt_lookup_successful = False
                
#                 # Strategy 1: Try with base domain name
#                 base_domain = domain_id.split('_')[0]
#                 alt_key = (base_domain, int(resid))
                
#                 if alt_key in self.rmsf_lookup:
#                     rmsf_value = self.rmsf_lookup[alt_key]
#                     alt_lookup_successful = True
#                     # Update the lookup for future use
#                     self.rmsf_lookup[lookup_key] = rmsf_value
                
#                 # If alternative lookup failed
#                 if not alt_lookup_successful:
#                     # Since we verified during initialization that this should have a match,
#                     # this is likely an error condition
#                     logger.warning(f"RMSF value not found for {domain_id}:{resid} (mapped to {rmsf_domain})")
                    
#                     # Return a dummy tensor with dataset mean
#                     rmsf_value = self.rmsf_data['average_rmsf'].median()
            
#             # Convert to tensors
#             voxel_tensor = torch.tensor(voxel, dtype=torch.float32)
#             rmsf_tensor = torch.tensor(rmsf_value, dtype=torch.float32)
            
#             if self.transform:
#                 voxel_tensor = self.transform(voxel_tensor)
                    
#             return voxel_tensor, rmsf_tensor
                
#         except Exception as e:
#             logger.error(f"Error retrieving item {idx} (domain={domain_id}, resid={resid}): {str(e)}")
            
#             # Create dummy data with same shape as expected to avoid crashing
#             if hasattr(self, '_dummy_shape'):
#                 dummy_shape = self._dummy_shape
#             else:
#                 # Try to find the shape from any valid sample
#                 for test_idx in range(min(100, len(self.samples))):
#                     if test_idx == idx:
#                         continue
#                     try:
#                         test_domain, test_resid = self.samples[test_idx]
#                         dummy_shape = self.voxel_data[test_domain][test_resid].shape
#                         self._dummy_shape = dummy_shape
#                         break
#                     except:
#                         continue
#                 else:
#                     # If we couldn't find any valid sample, use a default shape
#                     dummy_shape = (5, 24, 24, 24)  # Common shape for voxel data
#                     self._dummy_shape = dummy_shape
            
#             # Create dummy tensors
#             dummy_voxel = torch.zeros(dummy_shape, dtype=torch.float32)
#             dummy_rmsf = torch.tensor(0.0, dtype=torch.float32)
            
#             return dummy_voxel, dummy_rmsf
    
# def load_voxel_data(
#         voxel_file: str,
#         domain_ids: Optional[List[str]] = None,
#         max_domains: Optional[int] = None,
#         memory_ceiling_percent: float = 80.0,  # Reduced from original 85.0
#         out_of_core_mode: bool = True, 
#     ) -> Dict[str, Dict[str, np.ndarray]]:
#         """
#         Load voxel data from HDF5 file with improved memory management.
        
#         Args:
#             voxel_file: Path to HDF5 file
#             domain_ids: Optional list of domain IDs to load. If None, load all domains
#                         (subject to memory constraints)
#             max_domains: Optional limit on number of domains to load
#             memory_ceiling_percent: Memory usage ceiling (percent of system RAM)
#                                     above which no new domains will be loaded
#             out_of_core_mode: If True, use out-of-core processing with memory mapping
            
#         Returns:
#             Dictionary mapping domain IDs to dictionaries of residue voxel data
#         """
#         voxel_file = resolve_path(voxel_file)
#         logger.info(f"Loading voxel data from {voxel_file}")
        
#         if not os.path.exists(voxel_file):
#             raise FileNotFoundError(f"Voxel file not found: {voxel_file}")
        
#         # Get initial memory stats
#         memory_stats = check_memory_usage()
#         logger.info(f"Initial memory: System: {memory_stats['system_percent']}% used, "
#                 f"Process: {memory_stats['process_rss_gb']:.2f} GB")
        
#         # Aggressive memory clearing before starting to create space
#         clear_memory(force_gc=True, clear_cuda=True)
        
#         # Dictionary to store loaded domain data
#         voxel_data = {}
        
#         # Create temp directory for memory-mapped files if using out-of-core mode
#         temp_dir = None
#         if out_of_core_mode:
#             import tempfile
#             import atexit
#             import shutil
            
#             temp_dir = tempfile.mkdtemp(prefix="voxelflex_")
#             logger.info(f"Created temporary directory for out-of-core processing: {temp_dir}")
            
#             # Register cleanup function to ensure temp files are removed
#             def cleanup_temp_dir():
#                 if os.path.exists(temp_dir):
#                     logger.info(f"Removing temporary directory: {temp_dir}")
#                     try:
#                         shutil.rmtree(temp_dir)
#                     except Exception as e:
#                         logger.warning(f"Failed to remove temp directory: {str(e)}")
            
#             atexit.register(cleanup_temp_dir)
        
#         # Set a more conservative memory ceiling for loading new domains
#         # This leaves more headroom for processing after loading
#         actual_memory_ceiling = min(memory_ceiling_percent, 75.0)
#         logger.info(f"Using memory ceiling of {actual_memory_ceiling}% for domain loading")
        
#         with h5py.File(voxel_file, 'r') as f:
#             # Get list of all domains in the file
#             available_domains = list(f.keys())
#             logger.info(f"Found {len(available_domains)} domains in voxel file")
            
#             # Filter domains if domain_ids is provided and not empty
#             if domain_ids and len(domain_ids) > 0:
#                 logger.info(f"Filtering domains based on provided domain_ids (n={len(domain_ids)})")
#                 domains_to_process = []
#                 for domain in available_domains:
#                     # Extract base domain (remove _pdb, _pdb_clean, etc.)
#                     base_domain = domain.split('_')[0]
#                     if base_domain in domain_ids or domain in domain_ids:
#                         domains_to_process.append(domain)
                
#                 if not domains_to_process:
#                     logger.warning(f"None of the specified domain_ids were found in the voxel file")
#                     # Show some examples to help with debugging
#                     sample_domains = [d.split('_')[0] for d in available_domains[:10]]
#                     logger.debug(f"First few available domain IDs (base names): {sample_domains}...")
#                     logger.debug(f"Specified domain_ids: {domain_ids}")
                    
#                     # Try more flexible matching as a fallback
#                     logger.info("Trying more flexible domain matching...")
#                     for domain in available_domains:
#                         base_domain = domain.split('_')[0]
#                         # Try to find partial matches
#                         for domain_id in domain_ids:
#                             if domain_id in base_domain or base_domain in domain_id:
#                                 logger.info(f"Found flexible match: {domain} for requested {domain_id}")
#                                 domains_to_process.append(domain)
#                                 break
#             else:
#                 domains_to_process = available_domains
            
#             # Apply max_domains limit if specified
#             if max_domains is not None and max_domains > 0 and len(domains_to_process) > max_domains:
#                 logger.info(f"Limiting to {max_domains} domains (out of {len(domains_to_process)} available)")
#                 domains_to_process = domains_to_process[:max_domains]
            
#             # Analyze domains to estimate memory requirements and sort by size
#             domain_size_estimates = []
            
#             for domain_idx, domain_id in enumerate(domains_to_process[:min(50, len(domains_to_process))]):
#                 try:
#                     domain_group = f[domain_id]
                    
#                     if len(domain_group.keys()) == 0:
#                         logger.warning(f"Domain {domain_id} has no children")
#                         continue
                        
#                     first_child_key = list(domain_group.keys())[0]
#                     residue_group = domain_group[first_child_key]
                    
#                     # Count residues to estimate domain size
#                     residue_keys = [k for k in residue_group.keys() if isinstance(k, str) and k.isdigit()]
                    
#                     # If we have residues, estimate domain size
#                     if residue_keys:
#                         sample_residue = residue_keys[0]
#                         residue_data = residue_group[sample_residue]
                        
#                         if isinstance(residue_data, h5py.Dataset):
#                             # Estimate size of single residue
#                             shape = residue_data.shape
#                             element_size = residue_data.dtype.itemsize
#                             residue_size_bytes = np.prod(shape) * element_size
                            
#                             # Estimate size of domain based on residue count
#                             domain_size_bytes = residue_size_bytes * len(residue_keys)
#                             domain_size_gb = domain_size_bytes / (1024**3)
                            
#                             domain_size_estimates.append((domain_id, domain_size_gb, len(residue_keys)))
                            
#                             if domain_idx < 5 or domain_idx % 20 == 0:
#                                 logger.debug(f"Domain {domain_id}: ~{domain_size_gb:.4f} GB, {len(residue_keys)} residues")
#                 except Exception as e:
#                     logger.warning(f"Error estimating size for domain {domain_id}: {str(e)}")
            
#             # Sort domains by size (smaller domains first to prioritize)
#             if domain_size_estimates:
#                 # Sort by estimated size
#                 domain_size_estimates.sort(key=lambda x: x[1])
                
#                 # Reorder domains_to_process based on size
#                 sorted_domain_ids = [d[0] for d in domain_size_estimates]
#                 remaining_domain_ids = [d for d in domains_to_process if d not in sorted_domain_ids]
                
#                 domains_to_process = sorted_domain_ids + remaining_domain_ids
#                 logger.info(f"Prioritized domains by size. Smallest: {domain_size_estimates[0][0]} ({domain_size_estimates[0][1]:.4f} GB)")
#                 logger.info(f"Largest analyzed: {domain_size_estimates[-1][0]} ({domain_size_estimates[-1][1]:.4f} GB)")
            
#             # Calculate a more conservative domain limit based on memory
#             if domain_size_estimates:
#                 avg_domain_size_gb = sum(d[1] for d in domain_size_estimates) / len(domain_size_estimates)
#                 memory = psutil.virtual_memory()
#                 available_gb = memory.available / (1024**3)
                
#                 # More conservative memory usage (60% of available)
#                 safe_memory_usage_gb = available_gb * 0.6
                
#                 # Account for memory overhead and leave room for processing
#                 estimated_max_domains = int(safe_memory_usage_gb / (avg_domain_size_gb * 1.5))
                
#                 logger.info(f"Average domain size: {avg_domain_size_gb:.4f} GB")
#                 logger.info(f"Available memory: {available_gb:.2f} GB, safe usage: {safe_memory_usage_gb:.2f} GB")
#                 logger.info(f"Can safely process approximately {estimated_max_domains} domains")
                
#                 # Apply adaptive domain limit if needed
#                 if max_domains is None or estimated_max_domains < max_domains:
#                     domains_to_process = domains_to_process[:estimated_max_domains]
#                     logger.warning(f"Auto-limiting to {len(domains_to_process)} domains based on memory constraints")
            
#             # Process domains with careful memory monitoring
#             logger.info(f"Processing {len(domains_to_process)} domains")
#             domains_processed = 0
            
#             # Create progress bar
#             progress = EnhancedProgressBar(
#                 len(domains_to_process), 
#                 prefix="Loading domains", 
#                 suffix="Complete",
#                 stage_info="DATA_LOADING"
#             )
            
#             # Iterate through domains
#             for domain_idx, domain_id in enumerate(domains_to_process):
#                 # Check memory before loading this domain
#                 memory_stats = check_memory_usage()
#                 current_memory_percent = memory_stats['system_percent']
                
#                 # Use more conservative memory ceiling
#                 if current_memory_percent >= actual_memory_ceiling:
#                     logger.warning(f"Memory ceiling ({actual_memory_ceiling}%) reached at "
#                                 f"{current_memory_percent}% after processing {domains_processed} domains")
#                     logger.info(f"Stopping domain loading to preserve memory integrity")
#                     break
                
#                 # Process one complete domain
#                 try:
#                     domain_group = f[domain_id]
                    
#                     if len(domain_group.keys()) == 0:
#                         logger.warning(f"Domain {domain_id} has no children")
#                         progress.update(domain_idx + 1)
#                         continue
                        
#                     # Get the first direct child
#                     first_child_key = list(domain_group.keys())[0]
#                     residue_group = domain_group[first_child_key]
                    
#                     # Get all residue IDs (numeric keys)
#                     residue_keys = [k for k in residue_group.keys() if isinstance(k, str) and k.isdigit()]
                    
#                     if not residue_keys:
#                         logger.warning(f"No valid residues found for domain {domain_id}")
#                         progress.update(domain_idx + 1)
#                         continue
                    
#                     # Process all residues in the domain
#                     domain_data = {}
                    
#                     # Decide whether to use memory mapping
#                     use_memmap = out_of_core_mode and len(residue_keys) > 100  # Use memmap for larger domains
                    
#                     for resid in residue_keys:
#                         try:
#                             residue_data = residue_group[resid]
                            
#                             if isinstance(residue_data, h5py.Dataset):
#                                 # Get shape and determine if transpose is needed
#                                 shape = residue_data.shape
#                                 voxel_data_raw = residue_data[:]
                                
#                                 # For datasets with shape (x, y, z, channels), transpose to (channels, x, y, z)
#                                 if len(shape) == 4 and shape[3] in [4, 5]:
#                                     voxel = np.transpose(voxel_data_raw, (3, 0, 1, 2))
#                                 else:
#                                     voxel = voxel_data_raw
                                
#                                 # Convert to float32 if needed
#                                 if voxel.dtype == bool:
#                                     voxel = voxel.astype(np.float32)
#                                 elif voxel.dtype != np.float32:
#                                     voxel = voxel.astype(np.float32)
                                
#                                 # Store using memory mapping for larger domains
#                                 if use_memmap:
#                                     # Create memmap file
#                                     memmap_file = os.path.join(temp_dir, f"{domain_id}_{resid}.npy")
#                                     memmap_array = np.memmap(memmap_file, dtype=np.float32, 
#                                                         mode='w+', shape=voxel.shape)
#                                     # Copy data to memmap
#                                     memmap_array[:] = voxel[:]
#                                     memmap_array.flush()
                                    
#                                     # Store memmap in domain data
#                                     domain_data[resid] = memmap_array
#                                 else:
#                                     # Store smaller voxels directly
#                                     domain_data[resid] = voxel
                                
#                                 # Force explicit deletion of raw data
#                                 del voxel_data_raw
#                                 del voxel
#                         except Exception as e:
#                             logger.debug(f"Error processing residue {resid} for domain {domain_id}: {str(e)}")
                    
#                     # Store the domain data if it has content
#                     if domain_data:
#                         voxel_data[domain_id] = domain_data
#                         domains_processed += 1
                        
#                         # Log details for large domains
#                         if len(domain_data) > 200:
#                             logger.info(f"Loaded large domain {domain_id} with {len(domain_data)} residues")
#                     else:
#                         logger.warning(f"No valid voxel data found for domain {domain_id}")
                    
#                 except Exception as e:
#                     logger.error(f"Error processing domain {domain_id}: {str(e)}")
                
#                 # Update progress bar
#                 progress.update(domain_idx + 1)
                
#                 # Force periodic memory clearing
#                 if domain_idx % 2 == 0:  # More frequent clearing
#                     clear_memory(force_gc=True, clear_cuda=True)
            
#             # End progress bar
#             progress.finish()
            
#             # Report final processing statistics
#             total_domains = len(domains_to_process)
#             memory_stats = check_memory_usage()
#             current_memory_percent = memory_stats['system_percent']
            
#             logger.info(f"Domain loading complete: {domains_processed}/{total_domains} domains processed")
#             logger.info(f"Current memory usage: {current_memory_percent:.1f}% "
#                     f"({memory_stats['process_rss_gb']:.2f} GB)")
            
#             if domains_processed < total_domains:
#                 logger.warning(f"{total_domains - domains_processed} domains were not processed "
#                             f"due to memory constraints")
        
#         # Validate the loaded voxel data
#         if not voxel_data:
#             logger.error("No valid voxel data was loaded")
#             raise ValueError("No valid voxel data was loaded. Check the logs for details.")
        
#         logger.info(f"Successfully loaded {len(voxel_data)} domains with a total of "
#                 f"{sum(len(d) for d in voxel_data.values())} residues")
        
#         return voxel_data   
    

# # --- Start copy here ---
# def load_rmsf_data(
#     rmsf_dir: str,
#     replica: str = "replica_average", # Keep replica for filename generation consistency
#     temperature: Union[int, str] = 320
# ) -> pd.DataFrame:
#     """
#     Load RMSF data from CSV file.

#     Args:
#         rmsf_dir: Base directory containing the temperature-specific RMSF CSV files
#                   (e.g., /path/to/RMSF/replica_average)
#         replica: Replica identifier used in the filename (default: "replica_average")
#         temperature: Temperature value (default: 320)

#     Returns:
#         DataFrame containing RMSF data
#     """
#     rmsf_dir = resolve_path(rmsf_dir)
#     logger.info(f"Loading RMSF data from {rmsf_dir}, replica={replica}, temperature={temperature}")

#     if isinstance(temperature, int):
#         temperature_str = str(temperature)
#     else:
#         temperature_str = temperature

#     # --- CORRECTED PATH CONSTRUCTION ---
#     # Construct the filename directly, assuming rmsf_dir points to the correct location
#     filename = f"rmsf_{replica}_temperature{temperature_str}.csv"
#     rmsf_file = os.path.join(
#         rmsf_dir,
#         filename
#     )
#     # --- END CORRECTION ---

#     logger.info(f"Attempting to load RMSF file: {rmsf_file}") # Log the constructed path

#     if not os.path.exists(rmsf_file):
#         # Add more debugging info if file not found
#         logger.error(f"Directory listing for {rmsf_dir}: {os.listdir(rmsf_dir)}")
#         raise FileNotFoundError(f"RMSF file not found: {rmsf_file}")

#     # Load the CSV file
#     try:
#          rmsf_data = pd.read_csv(rmsf_file)
#     except Exception as e:
#          logger.error(f"Failed to read RMSF CSV file {rmsf_file}: {e}")
#          raise

#     # --- Add RMSF column renaming logic ---
#     expected_col = f"rmsf_{temperature_str}"
#     logger.debug(f"Checking for RMSF columns: expected='{expected_col}', alternative='average_rmsf'")
#     logger.debug(f"Columns found in CSV: {list(rmsf_data.columns)}")

#     if expected_col in rmsf_data.columns:
#         logger.info(f"Renaming column '{expected_col}' to 'average_rmsf' for processing.")
#         rmsf_data.rename(columns={expected_col: 'average_rmsf'}, inplace=True)
#     elif 'average_rmsf' not in rmsf_data.columns:
#         # If neither expected nor 'average_rmsf' exists, raise error
#         raise ValueError(f"Could not find required RMSF column '{expected_col}' or 'average_rmsf' in {rmsf_file}. Found columns: {list(rmsf_data.columns)}")
#     else:
#         logger.info("Column 'average_rmsf' already exists. Using it directly.")
#     # --- End RMSF column renaming ---

#     # Validate the RMSF data (now expecting 'average_rmsf')
#     rmsf_data = validate_rmsf_data(rmsf_data)

#     logger.info(f"Loaded RMSF data with {len(rmsf_data)} entries")
#     return rmsf_data
# # --- End copy here ---


# def create_domain_mapping(
#     voxel_domains: List[str],
#     rmsf_domains: List[str]
# ) -> Dict[str, str]:
#     """
#     Create mapping between voxel domain IDs and RMSF domain IDs with improved performance.
    
#     Args:
#         voxel_domains: List of domain IDs from voxel data
#         rmsf_domains: List of domain IDs from RMSF data
        
#     Returns:
#         Dictionary mapping voxel domain IDs to RMSF domain IDs
#     """
#     logger.info("Creating optimized domain mapping")
    
#     # OPTIMIZATION: Convert to sets for O(1) lookups
#     voxel_domains_set = set(voxel_domains)
#     rmsf_domains_set = set(rmsf_domains)
    
#     # Extract base names for all domains upfront
#     voxel_base_names = {d: d.split('_')[0] for d in voxel_domains}
#     rmsf_base_names = {d: d.split('_')[0] for d in rmsf_domains}
    
#     # Create reverse lookup (base name to full domain names)
#     rmsf_base_to_full = {}
#     for full_domain, base_name in rmsf_base_names.items():
#         if base_name not in rmsf_base_to_full:
#             rmsf_base_to_full[base_name] = []
#         rmsf_base_to_full[base_name].append(full_domain)
    
#     # Track suffix statistics
#     suffix_counts = {}
    
#     # Create mapping dictionary with optimal algorithm
#     mapping = {}
#     direct_matches = 0
#     base_matches = 0
#     fuzzy_matches = 0
    
#     # First, try direct matches (fastest)
#     for voxel_domain in voxel_domains:
#         if voxel_domain in rmsf_domains_set:
#             mapping[voxel_domain] = voxel_domain
#             direct_matches += 1
#         else:
#             # Track suffix for diagnostics
#             if "_" in voxel_domain:
#                 suffix = voxel_domain.split("_", 1)[1]
#                 suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    
#     # Next, try base name matches for unmatched domains
#     for voxel_domain in voxel_domains:
#         if voxel_domain not in mapping:
#             base_domain = voxel_base_names[voxel_domain]
            
#             # Check if base name is a direct match with any RMSF domain
#             if base_domain in rmsf_domains_set:
#                 mapping[voxel_domain] = base_domain
#                 base_matches += 1
#             # Check if base name matches any RMSF domain's base name
#             elif base_domain in rmsf_base_to_full:
#                 # Use the first RMSF domain with this base name
#                 mapping[voxel_domain] = rmsf_base_to_full[base_domain][0]
#                 base_matches += 1
    
#     # Finally, try fuzzy matching for any remaining unmatched domains
#     for voxel_domain in voxel_domains:
#         if voxel_domain not in mapping:
#             base_domain = voxel_base_names[voxel_domain]
            
#             # Try to find any partial match
#             for rmsf_domain in rmsf_domains:
#                 if rmsf_domain.startswith(base_domain) or base_domain.startswith(rmsf_domain):
#                     mapping[voxel_domain] = rmsf_domain
#                     fuzzy_matches += 1
#                     break
    
#     # Report mapping statistics
#     total_mapped = len(mapping)
#     mapping_percentage = (total_mapped / len(voxel_domains)) * 100 if voxel_domains else 0
    
#     logger.info(f"Created mapping for {total_mapped} domains out of {len(voxel_domains)} "
#                 f"({mapping_percentage:.1f}%)")
#     logger.info(f"Direct matches: {direct_matches}, Base matches: {base_matches}, Fuzzy matches: {fuzzy_matches}")
    
#     # Log suffix statistics
#     if suffix_counts:
#         # Get top 5 most common suffixes
#         top_suffixes = sorted(suffix_counts.items(), key=lambda x: x[1], reverse=True)[:5]
#         suffix_info = ", ".join([f"{suffix}: {count}" for suffix, count in top_suffixes])
#         logger.info(f"Domain suffix statistics (top 5): {suffix_info}")
        
#         # Get total count by suffix type
#         total_with_suffix = sum(suffix_counts.values())
#         suffix_percentage = 100.0 * total_with_suffix / len(voxel_domains)
#         logger.info(f"Domains with suffixes: {total_with_suffix} ({suffix_percentage:.1f}%)")
    
#     # If no mappings were created, provide more detailed information
#     if not mapping:
#         logger.error("No domain mappings could be created!")
#         logger.error("This could be due to naming inconsistencies between voxel and RMSF data")
#         logger.error(f"Sample voxel domains: {voxel_domains[:5]}")
#         logger.error(f"Sample RMSF domains: {rmsf_domains[:5]}")
        
#         # Try more permissive matching as a fallback
#         logger.info("Attempting more permissive matching as fallback...")
#         for voxel_domain in voxel_domains:
#             clean_voxel = voxel_domain.split('_')[0]
#             # Try to find any RMSF domain that starts with the same characters
#             for rmsf_domain in rmsf_domains:
#                 if rmsf_domain.startswith(clean_voxel) or clean_voxel.startswith(rmsf_domain):
#                     mapping[voxel_domain] = rmsf_domain
#                     logger.debug(f"Fallback mapping: {voxel_domain} -> {rmsf_domain}")
#                     break
        
#         if mapping:
#             logger.info(f"Created {len(mapping)} fallback mappings")
    
#     return mapping

# def prepare_dataloaders(
#     voxel_data: Dict[str, Dict[str, np.ndarray]],
#     rmsf_data: pd.DataFrame,
#     batch_size: int = 32,
#     train_split: float = 0.7,
#     val_split: float = 0.15,
#     test_split: float = 0.15,
#     num_workers: int = 4,
#     seed: int = 42,
#     safe_mode: bool = False,
#     memory_efficient: bool = True
# ) -> Tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Prepare PyTorch DataLoaders with aggressive memory management.
    
#     Args:
#         voxel_data: Dictionary mapping domain IDs to voxel data
#         rmsf_data: DataFrame containing RMSF values
#         batch_size: Batch size for DataLoaders
#         train_split: Proportion of data for training
#         val_split: Proportion of data for validation
#         test_split: Proportion of data for testing
#         num_workers: Number of worker processes for DataLoaders
#         seed: Random seed for reproducibility
#         safe_mode: If True, use single-threaded mode (no workers)
#         memory_efficient: If True, use memory-efficient dataset mode
        
#     Returns:
#         Tuple of (train_loader, val_loader, test_loader)
#     """
#     logger.info("Preparing DataLoaders")
    
#     # Check memory before dataset creation
#     memory_stats = check_memory_usage()
#     logger.info(f"Memory before dataloader creation: {memory_stats['system_percent']}% used")
    
#     # Force safe mode in extremely high memory situations
#     if memory_stats['system_percent'] > 90:
#         logger.warning(f"Critical memory state detected: {memory_stats['system_percent']}%. Forcing safe mode.")
#         safe_mode = True
        
#     # Adjust workers based on memory conditions
#     if not safe_mode:
#         # Use more aggressive worker reduction with thresholds
#         if memory_stats['system_percent'] > 85:
#             num_workers = 0  # Force single-threaded mode in high memory usage
#             logger.warning(f"High memory usage ({memory_stats['system_percent']}%). Forcing single-threaded mode.")
#         elif memory_stats['system_percent'] > 75:
#             num_workers = 1  # Use just one worker
#             logger.warning(f"Elevated memory usage ({memory_stats['system_percent']}%). Using 1 worker only.")
#         else:
#             adjusted_workers = min(2, multiprocessing.cpu_count() // 4)  # Much more conservative
#             if adjusted_workers != num_workers:
#                 logger.warning(f"Reducing workers from {num_workers} to {adjusted_workers} to conserve memory")
#                 num_workers = adjusted_workers
    
#     # Force single-threaded mode if safe_mode is enabled
#     if safe_mode:
#         old_workers = num_workers
#         num_workers = 0
#         logger.info(f"Safe mode enabled: Setting workers from {old_workers} to {num_workers}")
    
#     # Clear memory before dataset creation
#     clear_memory(force_gc=True, clear_cuda=True)
    
#     # Create domain mapping
#     voxel_domains = list(voxel_data.keys())
#     rmsf_domains = rmsf_data['domain_id'].unique().tolist()
#     domain_mapping = create_domain_mapping(voxel_domains, rmsf_domains)
    
#     # Validate the mapping
#     validate_domain_residue_mapping(voxel_data, rmsf_data, domain_mapping)
    
#     # Adjust batch size aggressively based on memory
#     input_shape = None
#     for domain_id, domain_data in voxel_data.items():
#         for resid, voxel in domain_data.items():
#             input_shape = voxel.shape
#             break
#         if input_shape is not None:
#             break
    
#     if input_shape is not None:
#         # Calculate element_size
#         element_size = 4  # float32 = 4 bytes
#         single_input_bytes = np.prod(input_shape) * element_size
        
#         # Calculate target memory usage (much more conservative)
#         memory_stats = check_memory_usage()
#         available_memory_bytes = memory_stats['system_available_gb'] * (1024**3)
        
#         # Target a smaller percentage of available memory (20-40% depending on memory pressure)
#         memory_target_percentage = 0.2 if memory_stats['system_percent'] > 85 else 0.4
#         target_memory_bytes = available_memory_bytes * memory_target_percentage
        
#         # Account for PyTorch overhead - assume 3x memory usage
#         pytorch_overhead = 3.0
#         # Increase overhead estimation in high memory conditions
#         if memory_stats['system_percent'] > 80:
#             pytorch_overhead = 4.0
        
#         # Calculate safe batch size
#         safe_batch_size = max(1, int(target_memory_bytes / (single_input_bytes * pytorch_overhead)))
        
#         # Cap batch size to a reasonable maximum to avoid memory spikes
#         max_safe_batch = 16 if memory_stats['system_percent'] < 80 else 8
#         safe_batch_size = min(safe_batch_size, max_safe_batch)
        
#         # If memory is critical, use an extremely small batch size
#         if memory_stats['system_percent'] > 90:
#             safe_batch_size = 1
        
#         # Apply the calculated safe batch size
#         if safe_batch_size < batch_size:
#             logger.warning(f"Reducing batch size from {batch_size} to {safe_batch_size} based on memory constraints")
#             batch_size = safe_batch_size
    
#     # Create dataset with memory-efficient mode
#     try:
#         logger.info("Creating RMSFDataset - this may take several minutes for large datasets")
        
#         # Force memory clearing before dataset creation
#         clear_memory(force_gc=True, clear_cuda=True)
        
#         with log_stage("DATASET_CREATION", "Creating PyTorch dataset"):
#             dataset = RMSFDataset(
#                 voxel_data, 
#                 rmsf_data, 
#                 domain_mapping, 
#                 memory_efficient=True  # Always use memory-efficient mode
#             )
#     except Exception as e:
#         logger.error(f"Error creating dataset: {str(e)}", exc_info=True)
#         raise
    
#     # Explicit memory clearing after dataset creation
#     clear_memory(force_gc=True, clear_cuda=True)
    
#     # Split the dataset with careful memory management
#     try:
#         dataset_size = len(dataset)
#         indices = list(range(dataset_size))
        
#         # Set a fixed seed for reproducibility
#         np.random.seed(seed)
#         np.random.shuffle(indices)
        
#         train_end = int(train_split * dataset_size)
#         val_end = train_end + int(val_split * dataset_size)
        
#         train_indices = indices[:train_end]
#         val_indices = indices[train_end:val_end]
#         test_indices = indices[val_end:]
        
#         logger.info(f"Split dataset into {len(train_indices)} training, "
#                     f"{len(val_indices)} validation, and {len(test_indices)} test samples")
        
#         # Force GC after creating large index lists
#         del indices
#         clear_memory(force_gc=True, clear_cuda=False)
#     except Exception as e:
#         logger.error(f"Error splitting dataset: {str(e)}", exc_info=True)
#         raise
    
#     # Configure DataLoader settings for extreme memory efficiency
#     pin_memory = False  # Disable pin_memory in all cases to reduce memory usage
#     persistent_workers = False  # Disable persistent workers to reduce memory usage
#     prefetch_factor = 2  # Use minimum prefetch factor
    
#     # Create DataLoaders with simplified configuration for memory efficiency
#     try:
#         logger.info(f"Creating DataLoaders with {num_workers} workers, minimal memory settings")
        
#         # Start with most conservative settings for all loaders
#         dataloader_kwargs = {
#             'batch_size': batch_size,
#             'num_workers': num_workers,
#             'pin_memory': False,
#             'persistent_workers': False,
#             'prefetch_factor': 2 if num_workers > 0 else None
#         }
        
#         # Create train loader with fallback
#         try:
#             train_loader = DataLoader(
#                 dataset, 
#                 sampler=torch.utils.data.SubsetRandomSampler(train_indices),
#                 **dataloader_kwargs
#             )
#         except Exception as e:
#             logger.error(f"Error creating train loader: {str(e)}. Falling back to minimal configuration.")
#             train_loader = DataLoader(
#                 dataset, 
#                 batch_size=1,  # Minimal batch size for extreme cases
#                 sampler=torch.utils.data.SubsetRandomSampler(train_indices),
#                 num_workers=0
#             )
        
#         # Force memory clearing between loader creation
#         clear_memory(force_gc=True, clear_cuda=True)
        
#         # Create validation loader with fallback
#         try:
#             val_loader = DataLoader(
#                 dataset,
#                 sampler=torch.utils.data.SubsetRandomSampler(val_indices),
#                 **dataloader_kwargs
#             )
#         except Exception as e:
#             logger.error(f"Error creating validation loader: {str(e)}. Falling back to minimal configuration.")
#             val_loader = DataLoader(
#                 dataset,
#                 batch_size=1,
#                 sampler=torch.utils.data.SubsetRandomSampler(val_indices),
#                 num_workers=0
#             )
        
#         # Force memory clearing between loader creation
#         clear_memory(force_gc=True, clear_cuda=True)
        
#         # Create test loader with fallback
#         try:
#             test_loader = DataLoader(
#                 dataset,
#                 sampler=torch.utils.data.SubsetRandomSampler(test_indices),
#                 **dataloader_kwargs
#             )
#         except Exception as e:
#             logger.error(f"Error creating test loader: {str(e)}. Falling back to minimal configuration.")
#             test_loader = DataLoader(
#                 dataset,
#                 batch_size=1,
#                 sampler=torch.utils.data.SubsetRandomSampler(test_indices),
#                 num_workers=0
#             )
        
#         # Verify loaders work by testing a single batch from test loader
#         try:
#             logger.info("Testing DataLoader initialization by accessing first batch...")
#             # Get first batch with timeout protection
#             import time
#             start_time = time.time()
#             timeout = 120  # 2 minutes timeout
            
#             for i, (inputs, targets) in enumerate(test_loader):
#                 logger.info(f"Successfully loaded first batch with {len(inputs)} items")
#                 break
                
#                 if time.time() - start_time > timeout:
#                     logger.warning("Timeout waiting for first batch. Loader may be stuck.")
#                     # Force break and try recovery
#                     break
            
#             logger.info("DataLoaders created and validated successfully")
#         except Exception as e:
#             logger.error(f"Error accessing first batch: {str(e)}")
            
#             # Final fallback to absolute minimal configuration
#             logger.warning("Falling back to absolute minimal DataLoader settings")
#             test_loader = DataLoader(
#                 dataset, 
#                 batch_size=1,
#                 sampler=torch.utils.data.SubsetRandomSampler(test_indices[:100]),  # Sample only 100 items
#                 num_workers=0,
#                 pin_memory=False
#             )
            
#             # Try again with even smaller subset if needed
#             try:
#                 next(iter(test_loader))
#                 logger.info("Successfully tested loader with minimal settings")
#             except:
#                 logger.critical("Failed to create working dataloader even with minimal settings!")
        
#         # Final memory clearing
#         clear_memory(force_gc=True, clear_cuda=True)
        
#         return train_loader, val_loader, test_loader
#     except Exception as e:
#         logger.error(f"Error creating DataLoaders: {str(e)}", exc_info=True)
#         raise
    
# def create_optimized_rmsf_lookup(rmsf_data: pd.DataFrame) -> Dict[Tuple[str, int], float]:
#     """
#     Create an optimized RMSF lookup dictionary that can be reused across dataset instances.
    
#     Args:
#         rmsf_data: DataFrame containing RMSF values
        
#     Returns:
#         Dictionary mapping (domain_id, resid) to RMSF values
#     """
#     logger.info("Creating global optimized RMSF lookup")
#     start_time = time.time()
    
#     # Extract columns as numpy arrays for fastest processing
#     domains = rmsf_data['domain_id'].values
#     resids = rmsf_data['resid'].values
#     rmsfs = rmsf_data['average_rmsf'].values
    
#     # Pre-allocate dictionary with size hint
#     rmsf_lookup = {}
    
#     # Process in a single pass for better performance
#     for i in range(len(domains)):
#         try:
#             resid_int = int(resids[i])
#             rmsf_lookup[(domains[i], resid_int)] = rmsfs[i]
#         except (ValueError, TypeError):
#             continue
    
#     # Also create base domain lookups
#     base_lookup = {}
#     for domain in set(domains):
#         base_domain = domain.split('_')[0] if '_' in domain else domain
#         if base_domain not in base_lookup:
#             base_lookup[base_domain] = {}
    
#     # Fill in base domain lookup
#     for (domain, resid), rmsf in rmsf_lookup.items():
#         base_domain = domain.split('_')[0] if '_' in domain else domain
#         base_lookup[base_domain][resid] = rmsf
    
#     # Add base domain lookups to main lookup
#     for base_domain, residues in base_lookup.items():
#         for resid, rmsf in residues.items():
#             if (base_domain, resid) not in rmsf_lookup:
#                 rmsf_lookup[(base_domain, resid)] = rmsf
    
#     lookup_time = time.time() - start_time
#     logger.info(f"Created global RMSF lookup with {len(rmsf_lookup)} entries in {lookup_time:.2f} seconds")
    
#     return rmsf_lookup

# def load_domain_batch(domain_indices, domain_list, config):
#     """
#     Load a batch of domains with optimized memory usage.
    
#     Args:
#         domain_indices: List of domain indices to load
#         domain_list: List of all domain IDs
#         config: Configuration dictionary
        
#     Returns:
#         Dictionary of loaded domain data
#     """
#     domain_batch = [domain_list[i] for i in domain_indices]
#     logger.info(f"Loading domain batch with {len(domain_batch)} domains")
    
#     # Load domains with a more efficient approach
#     voxel_data = {}
    
#     # Use larger read buffers for better I/O performance
#     read_buffer_size = 1024 * 1024 * 8  # 8MB buffer size
    
#     with h5py.File(config["input"]["voxel_file"], 'r', rdcc_nbytes=read_buffer_size) as f:
#         for domain_id in domain_batch:
#             try:
#                 domain_group = f[domain_id]
#                 first_child_key = list(domain_group.keys())[0]
#                 residue_group = domain_group[first_child_key]
#                 residue_keys = [k for k in residue_group.keys() if isinstance(k, str) and k.isdigit()]
                
#                 if not residue_keys:
#                     logger.warning(f"No valid residues found for domain {domain_id}")
#                     continue
                
#                 # Process all residues in the domain
#                 domain_data = {}
                
#                 # Pre-allocate dictionary with estimated size
#                 # This avoids dictionary resizing during insertion
#                 domain_data = {}
                
#                 # Process residues in larger chunks for better I/O
#                 chunk_size = 50
#                 for i in range(0, len(residue_keys), chunk_size):
#                     chunk = residue_keys[i:i+chunk_size]
#                     for resid in chunk:
#                         try:
#                             residue_data = residue_group[resid]
                            
#                             if isinstance(residue_data, h5py.Dataset):
#                                 # Get shape and determine if transpose is needed
#                                 shape = residue_data.shape
#                                 voxel_data_raw = residue_data[:]
                                
#                                 # For datasets with shape (x, y, z, channels), transpose to (channels, x, y, z)
#                                 if len(shape) == 4 and shape[3] in [4, 5]:
#                                     voxel = np.transpose(voxel_data_raw, (3, 0, 1, 2))
#                                 else:
#                                     voxel = voxel_data_raw
                                
#                                 # Convert to float32 if needed
#                                 if voxel.dtype == bool:
#                                     voxel = voxel.astype(np.float32)
#                                 elif voxel.dtype != np.float32:
#                                     voxel = voxel.astype(np.float32)
                                
#                                 domain_data[resid] = voxel
                                
#                                 # Force explicit deletion of raw data
#                                 del voxel_data_raw
                                
#                         except Exception as e:
#                             logger.debug(f"Error processing residue {resid} for domain {domain_id}: {str(e)}")
                
#                 # Store the domain data if it has content
#                 if domain_data:
#                     voxel_data[domain_id] = domain_data
#                 else:
#                     logger.warning(f"No valid voxel data found for domain {domain_id}")
                
#             except Exception as e:
#                 logger.error(f"Error processing domain {domain_id}: {str(e)}")
    
#     return voxel_data



# FILE: src/voxelflex/data/data_loader.py
"""
Data loading module for Voxelflex.

This module handles loading and processing of voxelized protein data (.hdf5)
and RMSF data (.csv).
"""

import os
import logging
import h5py
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import gc # Import gc for explicit collection
import shutil # For temp dir cleanup if using out_of_core
import atexit # For temp dir cleanup
import tempfile # For temp dir

import numpy as np
import pandas as pd
import torch
import multiprocessing
from torch.utils.data import Dataset, DataLoader

# Ensure these imports point to the correct modules in your project structure
from voxelflex.data.validators import (
    validate_voxel_data,
    validate_rmsf_data,
    validate_domain_residue_mapping
)
from voxelflex.utils.logging_utils import (
    get_logger,
    setup_logging, # Might not be needed here directly
    EnhancedProgressBar,
    log_memory_usage,
    log_operation_result,
    log_section_header,
    log_stage,
    log_step
)
from voxelflex.utils.file_utils import resolve_path
from voxelflex.utils.system_utils import (
    get_device, # Might not be needed here directly
    check_system_resources, # Might not be needed here directly
    clear_memory,
    check_memory_usage,
    set_num_threads, # Might not be needed here directly
    is_memory_critical,
    estimate_batch_size, # Might not be needed here directly
    adjust_workers_for_memory
)

logger = get_logger(__name__)

# --- RMSFDataset Class (No significant changes needed from previous optimization) ---
class RMSFDataset(Dataset):
    """PyTorch Dataset for voxel and RMSF data with dramatically improved performance."""

    def __init__(
        self,
        voxel_data: Dict[str, Dict[str, np.ndarray]],
        rmsf_data: pd.DataFrame,
        domain_mapping: Dict[str, str],
        transform=None,
        memory_efficient: bool = True, # Default to True for prediction/evaluation safety
        global_rmsf_lookup: Dict[Tuple[str, int], float] = None
    ):
        """
        Initialize RMSF dataset with highly optimized performance.
        """
        self.voxel_data = voxel_data
        self.domain_mapping = domain_mapping
        self.transform = transform
        self.memory_efficient = memory_efficient

        # Keep original DataFrame for __getitem__ fallbacks if needed
        self.rmsf_data_fallback_median = rmsf_data['average_rmsf'].median() if 'average_rmsf' in rmsf_data else 0.0


        start_time = time.time()

        if global_rmsf_lookup is not None:
            logger.debug("Using pre-computed global RMSF lookup")
            self.rmsf_lookup = global_rmsf_lookup
        else:
            logger.info("Creating optimized RMSF lookup dictionary for this dataset instance")
            self.rmsf_lookup = create_optimized_rmsf_lookup(rmsf_data) # Fallback if global not provided
            logger.info(f"Created local RMSF lookup in {time.time() - start_time:.2f} seconds")


        samples_start_time = time.time()
        logger.debug("Creating dataset samples list")

        # --- More Robust Sample Creation ---
        self.samples = []
        matched_residues = 0
        unmapped_residue_count = 0
        domain_sample_counts = {}

        for domain_id, domain_data in voxel_data.items():
            mapped_domain = domain_mapping.get(domain_id) # Use .get for safety
            count_in_domain = 0
            if mapped_domain is None:
                 # If a voxel domain has no mapping (should be rare after create_domain_mapping), skip its residues
                 unmapped_residue_count += len(domain_data)
                 continue

            for resid_str in domain_data:
                try:
                    resid_int = int(resid_str)
                    lookup_key = (mapped_domain, resid_int)

                    # Primary lookup using the mapped domain
                    if lookup_key in self.rmsf_lookup:
                        self.samples.append((domain_id, resid_str))
                        matched_residues += 1
                        count_in_domain +=1
                    else:
                         # Secondary lookup using potential base domain name from mapping keys
                         base_domain = mapped_domain.split('_')[0]
                         base_lookup_key = (base_domain, resid_int)
                         if base_domain != mapped_domain and base_lookup_key in self.rmsf_lookup:
                              self.samples.append((domain_id, resid_str))
                              matched_residues += 1
                              count_in_domain +=1
                         else:
                              unmapped_residue_count += 1
                              # logger.debug(f"Residue {domain_id}:{resid_str} could not be mapped to RMSF via {mapped_domain} or {base_domain}")


                except ValueError:
                    logger.warning(f"Invalid residue ID format: {resid_str} in domain {domain_id}")
                    unmapped_residue_count += 1
                except Exception as e:
                     logger.error(f"Unexpected error processing {domain_id}:{resid_str}: {e}")
                     unmapped_residue_count += 1
            domain_sample_counts[domain_id] = count_in_domain


        samples_time = time.time() - samples_start_time
        total_init_time = time.time() - start_time

        logger.info(f"Dataset initialized in {total_init_time:.2f}s. Samples created: {len(self.samples)} ({matched_residues} matched residues).")
        if unmapped_residue_count > 0:
             logger.warning(f"{unmapped_residue_count} residues from voxel data could not be matched to RMSF data.")
        # Log domains with zero samples if any
        zero_sample_domains = [d for d, count in domain_sample_counts.items() if count == 0]
        if zero_sample_domains:
            logger.warning(f"{len(zero_sample_domains)} domains had 0 matched residues: {zero_sample_domains[:5]}{'...' if len(zero_sample_domains)>5 else ''}")


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a data sample with improved error handling and lookup performance."""
        if idx >= len(self.samples):
             logger.error(f"Index {idx} out of bounds for dataset size {len(self.samples)}")
             # Return dummy data to prevent crash downstream
             return self._get_dummy_data()

        domain_id, resid_str = self.samples[idx]

        try:
            # Access voxel data safely
            voxel = self.voxel_data.get(domain_id, {}).get(resid_str)
            if voxel is None:
                 logger.error(f"Voxel data not found for sample {idx} ({domain_id}:{resid_str}) though it's in sample list.")
                 return self._get_dummy_data()


            # Get the corresponding RMSF value using optimized lookup
            rmsf_domain = self.domain_mapping.get(domain_id) # Should exist if in samples
            if rmsf_domain is None:
                logger.error(f"Domain mapping missing for {domain_id} in __getitem__ (Sample {idx})")
                return self._get_dummy_data()

            resid_int = int(resid_str) # Should be valid if in samples
            lookup_key = (rmsf_domain, resid_int)
            rmsf_value = self.rmsf_lookup.get(lookup_key)

            # Try base domain as fallback if direct lookup failed
            if rmsf_value is None:
                 base_domain = rmsf_domain.split('_')[0]
                 base_lookup_key = (base_domain, resid_int)
                 rmsf_value = self.rmsf_lookup.get(base_lookup_key)


            if rmsf_value is None:
                # This indicates an issue as the sample shouldn't exist if RMSF wasn't found during init
                logger.warning(f"RMSF value unexpectedly not found for {domain_id}:{resid_str} (mapped: {rmsf_domain}, base: {base_domain}). Using fallback median.")
                rmsf_value = self.rmsf_data_fallback_median


            # Convert to tensors
            # Ensure voxel data is numpy array before converting
            if not isinstance(voxel, np.ndarray):
                 # Handle memmap objects - read them into memory for the tensor
                 if hasattr(voxel, 'read'):
                      voxel_np = np.array(voxel)
                 else:
                      logger.error(f"Unexpected voxel data type for {domain_id}:{resid_str}: {type(voxel)}")
                      return self._get_dummy_data()
            else:
                 voxel_np = voxel

            voxel_tensor = torch.from_numpy(voxel_np.astype(np.float32)) # Ensure float32
            rmsf_tensor = torch.tensor(float(rmsf_value), dtype=torch.float32) # Ensure float

            if self.transform:
                voxel_tensor = self.transform(voxel_tensor)

            return voxel_tensor, rmsf_tensor

        except Exception as e:
            logger.error(f"Error retrieving item {idx} (domain={domain_id}, resid={resid_str}): {str(e)}")
            return self._get_dummy_data()

    def _get_dummy_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
         """Creates dummy data to return on error."""
         # Try to get shape dynamically, fallback to default
         dummy_shape = getattr(self, '_dummy_shape', (5, 24, 24, 24)) # Use default if not set
         if not hasattr(self, '_dummy_shape'):
              # Try to infer shape once
              try:
                   if self.samples:
                        sample_domain, sample_resid = self.samples[0]
                        dummy_shape = self.voxel_data[sample_domain][sample_resid].shape
                        self._dummy_shape = dummy_shape # Cache it
                   else: # No samples at all
                        logger.error("Dataset has no samples, cannot infer dummy shape.")
              except Exception:
                   logger.warning("Could not infer dummy shape, using default (5, 24, 24, 24).")

         dummy_voxel = torch.zeros(dummy_shape, dtype=torch.float32)
         dummy_rmsf = torch.tensor(0.0, dtype=torch.float32)
         return dummy_voxel, dummy_rmsf

# --- load_voxel_data function (No significant changes needed from previous optimization) ---
def load_voxel_data(
        voxel_file: str,
        domain_ids: Optional[List[str]] = None,
        max_domains: Optional[int] = None,
        memory_ceiling_percent: float = 80.0,
        out_of_core_mode: bool = True,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load voxel data from HDF5 file with improved memory management.
        (Implementation remains largely the same as previous version with memmap)
        """
        voxel_file = resolve_path(voxel_file)
        logger.info(f"Loading voxel data from {voxel_file}")

        if not os.path.exists(voxel_file):
            raise FileNotFoundError(f"Voxel file not found: {voxel_file}")

        memory_stats = check_memory_usage()
        logger.info(f"Initial memory before loading voxels: System: {memory_stats['system_percent']:.1f}% used, "
                    f"Process: {memory_stats['process_rss_gb']:.2f} GB")

        clear_memory(force_gc=True, clear_cuda=True) # Clear before starting

        voxel_data = {}
        temp_dir = None

        if out_of_core_mode:
            temp_dir = tempfile.mkdtemp(prefix="voxelflex_")
            logger.info(f"Using out-of-core mode. Temp directory: {temp_dir}")
            def cleanup_temp_dir():
                if temp_dir and os.path.exists(temp_dir):
                    logger.info(f"Cleaning up temporary directory: {temp_dir}")
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp directory {temp_dir}: {str(e)}")
            atexit.register(cleanup_temp_dir)

        # More conservative ceiling during loading phase
        actual_memory_ceiling = min(memory_ceiling_percent, 75.0)
        logger.info(f"Effective memory ceiling for loading: {actual_memory_ceiling}%")

        try:
            # Increase HDF5 buffer size for potentially faster reads
            with h5py.File(voxel_file, 'r', rdcc_nbytes=1024*1024*16) as f: # 16MB buffer
                available_domains = list(f.keys())
                logger.info(f"Found {len(available_domains)} total domains in HDF5 file.")

                # --- Domain Filtering Logic (same as predict.py) ---
                use_training_domains = False # This isn't typically needed for loading all data for train/val/test split
                domains_to_process = []

                if domain_ids and len(domain_ids) > 0:
                    logger.info(f"Filtering domains based on provided domain_ids list (n={len(domain_ids)}).")
                    available_set = set(available_domains)
                    for d_id in domain_ids:
                         base_d_id = d_id.split('_')[0]
                         found = False
                         for hdf5_domain in available_domains:
                             if hdf5_domain == d_id or hdf5_domain.split('_')[0] == base_d_id:
                                 if hdf5_domain not in domains_to_process: # Avoid duplicates
                                      domains_to_process.append(hdf5_domain)
                                 found = True
                                 # Don't break here, maybe multiple HDF5 domains match one base ID
                         if not found:
                              logger.warning(f"Specified domain_id '{d_id}' not found in HDF5 file.")
                    domains_to_process = list(set(domains_to_process)) # Ensure uniqueness after potential multi-matches
                    if not domains_to_process:
                        logger.error("None of the specified domain_ids were found. Cannot proceed.")
                        return {} # Return empty if no matches
                else:
                    logger.info("No specific domain_ids provided, considering all available domains.")
                    domains_to_process = available_domains

                logger.info(f"{len(domains_to_process)} domains selected initially based on domain_ids list.")
                # --- End Domain Filtering ---


                # Apply max_domains limit if specified
                if max_domains is not None and max_domains > 0:
                    if len(domains_to_process) > max_domains:
                        logger.info(f"Limiting to {max_domains} domains based on 'max_domains' config.")
                        # Maybe smarter selection later, for now just truncate
                        domains_to_process = domains_to_process[:max_domains]
                    else:
                         logger.info(f"'max_domains' ({max_domains}) is >= domains found ({len(domains_to_process)}), no limit applied.")


                # --- Domain Size Estimation and Sorting (Keep as is) ---
                domain_size_estimates = []
                logger.info("Estimating sizes for domain prioritization...")
                # Estimate for a sample to avoid iterating all
                sample_size_for_estimation = min(100, len(domains_to_process))
                for domain_idx, domain_id in enumerate(domains_to_process[:sample_size_for_estimation]):
                    try:
                        domain_group = f[domain_id]
                        if not domain_group: continue
                        first_child_key = list(domain_group.keys())[0]
                        residue_group = domain_group[first_child_key]
                        residue_keys = [k for k in residue_group.keys() if k.isdigit()]
                        if residue_keys:
                            sample_residue = residue_keys[0]
                            residue_data = residue_group[sample_residue]
                            if isinstance(residue_data, h5py.Dataset):
                                shape = residue_data.shape
                                element_size = residue_data.dtype.itemsize
                                residue_size_bytes = np.prod(shape) * element_size
                                domain_size_bytes = residue_size_bytes * len(residue_keys)
                                domain_size_gb = domain_size_bytes / (1024**3)
                                domain_size_estimates.append((domain_id, domain_size_gb, len(residue_keys)))
                                if domain_idx < 5 or domain_idx % 20 == 0:
                                    logger.debug(f"Est. Size: {domain_id}: ~{domain_size_gb:.4f} GB ({len(residue_keys)} res)")
                    except Exception as e:
                        logger.debug(f"Skipping size estimation for {domain_id}: {str(e)}")

                if domain_size_estimates:
                    domain_size_estimates.sort(key=lambda x: x[1]) # Sort smallest first
                    sorted_domain_ids = [d[0] for d in domain_size_estimates]
                    remaining_domain_ids = [d for d in domains_to_process if d not in sorted_domain_ids]
                    domains_to_process = sorted_domain_ids + remaining_domain_ids
                    logger.info(f"Prioritized {len(sorted_domain_ids)} domains by estimated size.")
                # --- End Domain Size Estimation ---


                # --- Memory-based Domain Limit (Keep as is, maybe adjust factor) ---
                if domain_size_estimates:
                    avg_domain_size_gb = sum(d[1] for d in domain_size_estimates) / len(domain_size_estimates)
                    memory = psutil.virtual_memory()
                    available_gb = memory.available / (1024**3)
                    # Use a slightly more generous factor for system RAM vs GPU RAM estimation
                    safe_memory_usage_gb = available_gb * 0.65 # Use up to 65% of available RAM
                    # Overhead factor for Python objects, dicts etc.
                    overhead_factor = 1.75 # Adjust based on observation
                    estimated_max_domains = int(safe_memory_usage_gb / (avg_domain_size_gb * overhead_factor)) if avg_domain_size_gb > 0 else len(domains_to_process)

                    logger.info(f"Avg. est. domain size: {avg_domain_size_gb:.4f} GB. Available RAM: {available_gb:.2f} GB.")
                    logger.info(f"Est. max domains based on 65% available RAM & {overhead_factor}x overhead: {estimated_max_domains}")

                    # Apply adaptive domain limit ONLY if no specific max_domains was set
                    if max_domains is None and estimated_max_domains < len(domains_to_process):
                        logger.warning(f"Auto-limiting to {estimated_max_domains} domains based on memory estimation.")
                        domains_to_process = domains_to_process[:estimated_max_domains]
                # --- End Memory-based Limit ---


                logger.info(f"Attempting to load data for {len(domains_to_process)} domains.")
                domains_processed_count = 0
                progress = EnhancedProgressBar(
                    len(domains_to_process),
                    prefix="Loading domains",
                    suffix="Complete",
                    stage_info="VOXEL_LOAD"
                )

                # --- Domain Loading Loop (Keep memmap logic) ---
                for domain_idx, domain_id in enumerate(domains_to_process):
                    memory_stats = check_memory_usage()
                    if memory_stats['system_percent'] >= actual_memory_ceiling:
                        logger.warning(f"Memory ceiling ({actual_memory_ceiling}%) reached. Stopping domain loading.")
                        break

                    try:
                        domain_group = f[domain_id]
                        if not domain_group: continue
                        first_child_key = list(domain_group.keys())[0]
                        residue_group = domain_group[first_child_key]
                        residue_keys = [k for k in residue_group.keys() if k.isdigit()]
                        if not residue_keys: continue

                        domain_data = {}
                        # Use memmap for larger domains if out_of_core_mode is True
                        use_memmap = out_of_core_mode and len(residue_keys) > 150 # Threshold for using memmap

                        for resid in residue_keys:
                            try:
                                residue_dataset = residue_group[resid]
                                if isinstance(residue_dataset, h5py.Dataset):
                                    shape = residue_dataset.shape
                                    voxel_data_raw = residue_dataset[:] # Read data

                                    # Transpose if necessary
                                    if len(shape) == 4 and shape[3] in [4, 5]:
                                        voxel = np.transpose(voxel_data_raw, (3, 0, 1, 2))
                                    else:
                                        voxel = voxel_data_raw

                                    # Ensure float32
                                    voxel = voxel.astype(np.float32, copy=False) # Avoid copy if possible

                                    if use_memmap and temp_dir:
                                        memmap_file = os.path.join(temp_dir, f"{domain_id}_{resid}.npy")
                                        memmap_array = np.memmap(memmap_file, dtype=np.float32,
                                                                 mode='w+', shape=voxel.shape)
                                        memmap_array[:] = voxel[:]
                                        memmap_array.flush()
                                        domain_data[resid] = memmap_array
                                        del memmap_array # Release handle, data is on disk
                                    else:
                                        domain_data[resid] = voxel # Store directly in memory

                                    del voxel_data_raw, voxel # Explicit cleanup

                            except Exception as res_e:
                                logger.debug(f"Error processing residue {domain_id}:{resid}: {res_e}")

                        if domain_data:
                            voxel_data[domain_id] = domain_data
                            domains_processed_count += 1
                        else:
                             logger.warning(f"No valid residues loaded for domain {domain_id}")


                    except Exception as domain_e:
                        logger.error(f"Failed to process domain {domain_id}: {domain_e}")

                    progress.update(domain_idx + 1)

                    # Periodic cleanup
                    if domain_idx % 5 == 0: # More frequent cleanup
                        clear_memory(force_gc=True, clear_cuda=False) # Don't clear CUDA cache here
                # --- End Domain Loading Loop ---

                progress.finish()
                logger.info(f"Domain loading finished. {domains_processed_count}/{len(domains_to_process)} domains loaded.")
                if domains_processed_count < len(domains_to_process):
                     logger.warning(f"{len(domains_to_process) - domains_processed_count} domains skipped, likely due to memory limits.")

        except Exception as file_e:
             logger.error(f"Failed to open or process HDF5 file {voxel_file}: {file_e}")
             raise # Re-raise after logging


        if not voxel_data:
            logger.error("No voxel data was loaded. Check HDF5 file structure, domain IDs, and memory limits.")
            raise ValueError("No voxel data loaded.")

        # Validation is good, but can consume extra memory. Maybe make optional?
        # logger.info("Validating loaded voxel data...")
        # try:
        #     voxel_data = validate_voxel_data(voxel_data)
        # except ValueError as val_err:
        #      logger.error(f"Voxel data validation failed: {val_err}")
        #      # Decide whether to proceed with potentially invalid data or raise
        #      # raise val_err # Option: Stop if validation fails
        # logger.info("Voxel data validation complete.")


        final_memory_stats = check_memory_usage()
        logger.info(f"Memory after loading voxels: System: {final_memory_stats['system_percent']:.1f}% used, "
                    f"Process: {final_memory_stats['process_rss_gb']:.2f} GB")
        logger.info(f"Successfully loaded data for {len(voxel_data)} domains.")

        return voxel_data


# --- load_rmsf_data (Corrected version from previous step) ---
def load_rmsf_data(
    rmsf_dir: str,
    replica: str = "replica_average",
    temperature: Union[int, str] = 320
) -> pd.DataFrame:
    """Loads RMSF data, handling path construction and column renaming."""
    rmsf_dir = resolve_path(rmsf_dir)
    logger.info(f"Loading RMSF data from {rmsf_dir}, replica={replica}, temperature={temperature}")

    if isinstance(temperature, int):
        temperature_str = str(temperature)
    else:
        temperature_str = temperature

    filename = f"rmsf_{replica}_temperature{temperature_str}.csv"
    rmsf_file = os.path.join(rmsf_dir, filename)

    logger.info(f"Attempting to load RMSF file: {rmsf_file}")

    if not os.path.exists(rmsf_file):
        try:
            logger.error(f"Directory listing for {rmsf_dir}: {os.listdir(rmsf_dir)}")
        except OSError as list_err:
            logger.error(f"Could not list directory {rmsf_dir}: {list_err}")
        raise FileNotFoundError(f"RMSF file not found: {rmsf_file}")

    try:
         rmsf_data = pd.read_csv(rmsf_file)
    except Exception as e:
         logger.error(f"Failed to read RMSF CSV file {rmsf_file}: {e}")
         raise

    expected_col = f"rmsf_{temperature_str}"
    logger.debug(f"Checking for RMSF columns: expected='{expected_col}', alternative='average_rmsf'")
    logger.debug(f"Columns found in CSV: {list(rmsf_data.columns)}")

    if expected_col in rmsf_data.columns:
        logger.info(f"Renaming column '{expected_col}' to 'average_rmsf' for processing.")
        rmsf_data.rename(columns={expected_col: 'average_rmsf'}, inplace=True)
    elif 'average_rmsf' not in rmsf_data.columns:
        raise ValueError(f"Could not find required RMSF column '{expected_col}' or 'average_rmsf' in {rmsf_file}. Found columns: {list(rmsf_data.columns)}")
    else:
        logger.info("Column 'average_rmsf' already exists. Using it directly.")

    rmsf_data = validate_rmsf_data(rmsf_data) # Validate after renaming

    logger.info(f"Loaded and validated RMSF data with {len(rmsf_data)} entries.")
    return rmsf_data


# --- create_domain_mapping (No significant changes needed from previous optimization) ---
def create_domain_mapping(
    voxel_domains: List[str],
    rmsf_domains: List[str]
) -> Dict[str, str]:
    """Creates mapping between voxel and RMSF domain IDs."""
    logger.info("Creating optimized domain mapping")
    voxel_domains_set = set(voxel_domains)
    rmsf_domains_set = set(rmsf_domains)
    voxel_base_names = {d: d.split('_')[0] for d in voxel_domains}
    rmsf_base_names = {d: d.split('_')[0] for d in rmsf_domains}

    rmsf_base_to_full = {}
    for full_domain, base_name in rmsf_base_names.items():
        if base_name not in rmsf_base_to_full:
            rmsf_base_to_full[base_name] = []
        rmsf_base_to_full[base_name].append(full_domain)

    mapping = {}
    direct_matches = 0
    base_matches = 0
    fuzzy_matches = 0

    # Direct Match
    for voxel_domain in voxel_domains:
        if voxel_domain in rmsf_domains_set:
            mapping[voxel_domain] = voxel_domain
            direct_matches += 1

    # Base Name Match
    for voxel_domain in voxel_domains:
        if voxel_domain not in mapping:
            base_domain = voxel_base_names[voxel_domain]
            if base_domain in rmsf_domains_set:
                mapping[voxel_domain] = base_domain
                base_matches += 1
            elif base_domain in rmsf_base_to_full:
                # Prefer shorter RMSF name if multiple matches for a base name
                potential_matches = sorted(rmsf_base_to_full[base_domain], key=len)
                mapping[voxel_domain] = potential_matches[0]
                base_matches += 1

    # Fuzzy Match (Less reliable, use as last resort - maybe remove?)
    # Consider removing fuzzy matching if it causes incorrect mappings.
    # for voxel_domain in voxel_domains:
    #     if voxel_domain not in mapping:
    #         base_domain = voxel_base_names[voxel_domain]
    #         for rmsf_domain in rmsf_domains:
    #             rmsf_base = rmsf_base_names[rmsf_domain]
    #             # Only match if base names are reasonably close
    #             if (rmsf_domain.startswith(base_domain) or base_domain.startswith(rmsf_domain)) and \
    #                (len(rmsf_base) > 3 and len(base_domain) > 3): # Avoid trivial matches
    #                 mapping[voxel_domain] = rmsf_domain
    #                 fuzzy_matches += 1
    #                 logger.debug(f"Fuzzy match: {voxel_domain} -> {rmsf_domain}")
    #                 break # Take first fuzzy match

    total_mapped = len(mapping)
    mapping_percentage = (total_mapped / len(voxel_domains)) * 100 if voxel_domains else 0
    logger.info(f"Created mapping for {total_mapped}/{len(voxel_domains)} voxel domains ({mapping_percentage:.1f}%).")
    logger.info(f"Matches: Direct={direct_matches}, BaseName={base_matches}, Fuzzy={fuzzy_matches}")

    if not mapping and voxel_domains:
        logger.error("No domain mappings created. Check naming conventions (e.g., '1abcA00' vs '1abcA00_pdb')")
        logger.error(f"Sample Voxel Domains: {voxel_domains[:5]}...")
        logger.error(f"Sample RMSF Domains: {rmsf_domains[:5]}...")

    return mapping


# --- prepare_dataloaders (No significant changes needed from previous optimization) ---
# This function creates the train/val/test split and corresponding DataLoaders
# It uses the RMSFDataset internally. Its memory management was already aggressive.
def prepare_dataloaders(
    voxel_data: Dict[str, Dict[str, np.ndarray]],
    rmsf_data: pd.DataFrame,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
    safe_mode: bool = False,
    memory_efficient: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare PyTorch DataLoaders with memory management."""
    logger.info("Preparing DataLoaders for training/validation/testing...")

    memory_stats = check_memory_usage()
    logger.info(f"Memory before DataLoader prep: System {memory_stats['system_percent']:.1f}%")

    # Adjust workers based on memory
    num_workers = adjust_workers_for_memory(num_workers)
    if safe_mode:
        logger.info("Safe mode enabled, forcing num_workers = 0.")
        num_workers = 0

    clear_memory(force_gc=True, clear_cuda=False) # Clear host memory

    # Create domain mapping
    voxel_domains = list(voxel_data.keys())
    rmsf_domains = rmsf_data['domain_id'].unique().tolist()
    domain_mapping = create_domain_mapping(voxel_domains, rmsf_domains)
    validate_domain_residue_mapping(voxel_data, rmsf_data, domain_mapping)

    # --- Create the Full Dataset Instance ---
    logger.info("Creating full RMSFDataset instance...")
    global_lookup = create_optimized_rmsf_lookup(rmsf_data) # Create lookup once
    try:
        dataset = RMSFDataset(
            voxel_data,
            rmsf_data,
            domain_mapping,
            memory_efficient=memory_efficient, # Use config setting
            global_rmsf_lookup=global_lookup
        )
        if len(dataset) == 0:
             raise ValueError("Created dataset is empty. Check data matching and logs.")
    except Exception as e:
        logger.error(f"Fatal error creating RMSFDataset: {e}", exc_info=True)
        raise

    logger.info(f"Full dataset created with {len(dataset)} samples.")
    clear_memory(force_gc=True, clear_cuda=False) # Clear potential intermediates


    # --- Split Dataset Indices ---
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)

    if not (0.99 < train_split + val_split + test_split < 1.01):
         logger.warning(f"Train/Val/Test splits ({train_split}, {val_split}, {test_split}) do not sum to 1. Adjusting test split.")
         test_split = 1.0 - train_split - val_split

    train_end = int(train_split * dataset_size)
    val_end = train_end + int(val_split * dataset_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    logger.info(f"Dataset split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    del indices # Free memory
    gc.collect()


    # --- Create DataLoaders ---
    # Settings for potentially better speed if memory allows
    pin_memory_setting = (torch.cuda.is_available()) and (memory_stats['system_percent'] < MEMORY_CRITICAL_THRESHOLD * 100)
    prefetch_factor_setting = 4 if num_workers > 0 else None # Increase prefetch

    dataloader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory_setting,
        'persistent_workers': False, # Keep False for most cases unless dataset is huge and static
        'prefetch_factor': prefetch_factor_setting
    }
    logger.info(f"DataLoader settings: batch_size={batch_size}, workers={num_workers}, pin_memory={pin_memory_setting}, prefetch={prefetch_factor_setting}")


    # Function to create loader with fallback
    def create_loader(subset_indices, shuffle=False, current_batch_size=batch_size):
        sampler = torch.utils.data.SubsetRandomSampler(subset_indices) if shuffle else torch.utils.data.SequentialSampler(subset_indices)
        try:
            return DataLoader(dataset, batch_size=current_batch_size, sampler=sampler, **dataloader_kwargs)
        except Exception as e:
            logger.error(f"Error creating DataLoader (batch={current_batch_size}, workers={num_workers}): {e}. Retrying with minimal settings.")
            clear_memory(force_gc=True, clear_cuda=True) # Aggressive clear on error
            try:
                 # Fallback 1: No workers, smaller batch
                 fallback_batch_size = max(1, current_batch_size // 4)
                 logger.warning(f"Retrying DataLoader creation with batch_size={fallback_batch_size}, num_workers=0")
                 return DataLoader(dataset, batch_size=fallback_batch_size, sampler=sampler, num_workers=0, pin_memory=False)
            except Exception as e2:
                 # Fallback 2: Batch size 1
                 logger.error(f"Error creating fallback DataLoader: {e2}. Retrying with batch_size=1.")
                 return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, pin_memory=False)


    logger.info("Creating train DataLoader...")
    train_loader = create_loader(train_indices, shuffle=True, current_batch_size=batch_size)
    clear_memory(force_gc=True, clear_cuda=False)

    logger.info("Creating validation DataLoader...")
    # Use potentially smaller batch size for validation to save memory if needed
    val_batch_size = max(1, batch_size // 2)
    val_loader = create_loader(val_indices, shuffle=False, current_batch_size=val_batch_size)
    clear_memory(force_gc=True, clear_cuda=False)

    logger.info("Creating test DataLoader...")
    test_batch_size = max(1, batch_size // 2)
    test_loader = create_loader(test_indices, shuffle=False, current_batch_size=test_batch_size)
    clear_memory(force_gc=True, clear_cuda=False)


    # Test loaders if possible
    try:
        logger.info("Quickly testing DataLoaders...")
        _ = next(iter(train_loader))
        _ = next(iter(val_loader))
        _ = next(iter(test_loader))
        logger.info("DataLoaders seem operational.")
    except Exception as test_e:
        logger.warning(f"Could not fetch first batch during DataLoader test: {test_e}")


    final_memory_stats = check_memory_usage()
    logger.info(f"Memory after DataLoader prep: System {final_memory_stats['system_percent']:.1f}%")

    return train_loader, val_loader, test_loader


# --- create_optimized_rmsf_lookup (No changes needed) ---
def create_optimized_rmsf_lookup(rmsf_data: pd.DataFrame) -> Dict[Tuple[str, int], float]:
    """Creates an optimized RMSF lookup dictionary."""
    logger.info("Creating global optimized RMSF lookup")
    start_time = time.time()
    domains = rmsf_data['domain_id'].values
    resids = rmsf_data['resid'].values
    rmsfs = rmsf_data['average_rmsf'].values

    rmsf_lookup = {}
    for i in range(len(domains)):
        try:
            resid_int = int(resids[i])
            rmsf_lookup[(domains[i], resid_int)] = rmsfs[i]
        except (ValueError, TypeError): continue # Skip invalid entries

    # Add base domain lookups for flexibility
    base_domain_map = {domain: domain.split('_')[0] for domain in set(domains)}
    for (domain, resid), rmsf_val in list(rmsf_lookup.items()): # Iterate over copy
         base_domain = base_domain_map.get(domain)
         if base_domain and base_domain != domain:
              base_key = (base_domain, resid)
              if base_key not in rmsf_lookup: # Add if base key doesn't already exist
                   rmsf_lookup[base_key] = rmsf_val


    lookup_time = time.time() - start_time
    logger.info(f"Global RMSF lookup created ({len(rmsf_lookup)} entries) in {lookup_time:.2f}s")
    return rmsf_lookup


# --- load_domain_batch (No significant changes needed from previous optimization) ---
def load_domain_batch(domain_indices: List[int], domain_list: List[str], config: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    """Loads a batch of domains with optimized memory usage."""
    domain_batch_ids = [domain_list[i] for i in domain_indices]
    logger.info(f"Loading domain batch with {len(domain_batch_ids)} domains")
    voxel_data = {}
    voxel_file = resolve_path(config["input"]["voxel_file"])
    read_buffer_size = 1024 * 1024 * 16 # 16MB HDF5 buffer

    temp_dir = None
    out_of_core_mode = config.get("system_utilization", {}).get("out_of_core_mode", False)
    if out_of_core_mode:
         # Create temp dir only if needed and not already existing (handle external calls)
         # This part needs careful management if called repeatedly outside load_voxel_data
         # For simplicity here, assume it's created if needed, but real use might need context manager
         pass # Rely on load_voxel_data's temp dir if called within pipeline


    try:
        with h5py.File(voxel_file, 'r', rdcc_nbytes=read_buffer_size) as f:
            for domain_id in domain_batch_ids:
                try:
                    if domain_id not in f:
                         logger.warning(f"Domain {domain_id} requested in batch not found in HDF5 file.")
                         continue
                    domain_group = f[domain_id]
                    if not domain_group: continue
                    first_child_key = list(domain_group.keys())[0]
                    residue_group = domain_group[first_child_key]
                    residue_keys = [k for k in residue_group.keys() if k.isdigit()]
                    if not residue_keys: continue

                    domain_data = {}
                    use_memmap = out_of_core_mode and len(residue_keys) > 150 and temp_dir is not None

                    for resid in residue_keys:
                        try:
                            residue_dataset = residue_group[resid]
                            if isinstance(residue_dataset, h5py.Dataset):
                                shape = residue_dataset.shape
                                voxel_data_raw = residue_dataset[:]

                                if len(shape) == 4 and shape[3] in [4, 5]:
                                    voxel = np.transpose(voxel_data_raw, (3, 0, 1, 2))
                                else:
                                    voxel = voxel_data_raw

                                voxel = voxel.astype(np.float32, copy=False)

                                if use_memmap:
                                     # This assumes temp_dir exists and is correct path
                                     # Proper implementation might pass temp_dir as argument
                                     # memmap_file = os.path.join(temp_dir, f"{domain_id}_{resid}.npy")
                                     # memmap_array = np.memmap(memmap_file, ...)
                                     # domain_data[resid] = memmap_array
                                     # For now, keep in memory if temp_dir isn't handled here
                                     domain_data[resid] = voxel
                                else:
                                    domain_data[resid] = voxel

                                del voxel_data_raw # Explicit cleanup
                                if 'memmap_array' in locals(): del memmap_array

                        except Exception as res_e:
                            logger.debug(f"Error loading residue {domain_id}:{resid}: {res_e}")

                    if domain_data:
                        voxel_data[domain_id] = domain_data

                except Exception as domain_e:
                    logger.error(f"Error processing domain {domain_id} in batch load: {domain_e}")
    except Exception as file_e:
         logger.error(f"Could not open HDF5 file {voxel_file} for batch loading: {file_e}")
         # Return empty or raise, depending on desired behavior
         return {}


    logger.info(f"Loaded data for {len(voxel_data)} domains in batch.")
    return voxel_data



class PredictionRMSFDataset(Dataset):
    """
    Dataset for prediction that loads voxel data on demand from HDF5.
    """
    def __init__(
        self,
        sample_list: List[Tuple[str, str]], # List of (domain_id, resid_str)
        voxel_file_path: str,
        domain_mapping: Dict[str, str],
        rmsf_lookup: Dict[Tuple[str, int], float],
        fallback_rmsf: float,
        # Store expected shape to create dummy data on error
        input_shape: Optional[Tuple[int, ...]] = (5, 24, 24, 24) # Default shape
    ):
        self.samples = sample_list
        self.voxel_file_path = voxel_file_path
        self.domain_mapping = domain_mapping
        self.rmsf_lookup = rmsf_lookup
        self.fallback_rmsf = fallback_rmsf
        self.input_shape = input_shape # Store for dummy data generation
        self.h5_child_key_cache = {} # Cache the child key (e.g., 'PDB') per domain

        logger.info(f"PredictionRMSFDataset initialized with {len(self.samples)} samples.")
        if not self.samples:
            logger.warning("PredictionRMSFDataset initialized with zero samples!")

    def _get_h5_child_key(self, f: h5py.File, domain_id: str) -> Optional[str]:
        """Finds the immediate child group containing residues (e.g., 'PDB'). Caches result."""
        if domain_id in self.h5_child_key_cache:
            return self.h5_child_key_cache[domain_id]

        if domain_id not in f:
            logger.error(f"Domain {domain_id} not found in HDF5 file during child key lookup.")
            return None

        domain_group = f[domain_id]
        if not domain_group: return None

        # Assume the first group child contains the residues
        for key in domain_group.keys():
            if isinstance(domain_group[key], h5py.Group):
                 # Basic check: does it contain numeric keys?
                 sub_keys = list(domain_group[key].keys())
                 if any(k.isdigit() for k in sub_keys):
                      self.h5_child_key_cache[domain_id] = key
                      return key

        logger.warning(f"Could not determine residue group key for domain {domain_id}. Keys: {list(domain_group.keys())}")
        self.h5_child_key_cache[domain_id] = None # Cache failure
        return None


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self.samples):
            logger.error(f"Index {idx} out of bounds for dataset size {len(self.samples)}")
            return self._get_dummy_data()

        domain_id, resid_str = self.samples[idx]

        try:
            # --- Load Voxel On Demand ---
            voxel_np = None
            # Open HDF5 file for each item - necessary for multiprocessing safety with standard h5py
            # OS caching should mitigate performance impact significantly.
            with h5py.File(self.voxel_file_path, 'r') as f:
                child_key = self._get_h5_child_key(f, domain_id)
                if child_key is None:
                     raise KeyError(f"Could not find residue group for {domain_id}")

                residue_dataset = f[domain_id][child_key][resid_str]
                voxel_data_raw = residue_dataset[:] # Read data

            # Transpose if necessary
            shape = voxel_data_raw.shape
            if len(shape) == 4 and shape[3] in [4, 5]:
                voxel_np = np.transpose(voxel_data_raw, (3, 0, 1, 2)).astype(np.float32, copy=False)
            else:
                voxel_np = voxel_data_raw.astype(np.float32, copy=False)

            del voxel_data_raw # Explicit cleanup
            # --- End Voxel Loading ---

            # --- RMSF Lookup (same logic as RMSFDataset) ---
            rmsf_domain = self.domain_mapping.get(domain_id)
            if rmsf_domain is None:
                 # Should not happen if samples were generated correctly
                 logger.error(f"Domain mapping unexpectedly missing for {domain_id} in prediction dataset.")
                 rmsf_value = self.fallback_rmsf
            else:
                try:
                    resid_int = int(resid_str)
                    lookup_key = (rmsf_domain, resid_int)
                    rmsf_value = self.rmsf_lookup.get(lookup_key)

                    if rmsf_value is None: # Try base domain fallback
                        base_domain = rmsf_domain.split('_')[0]
                        base_lookup_key = (base_domain, resid_int)
                        rmsf_value = self.rmsf_lookup.get(base_lookup_key)

                    if rmsf_value is None: # Still not found? Use fallback.
                        logger.debug(f"RMSF value still not found for {domain_id}:{resid_str} (mapped: {rmsf_domain}). Using fallback.")
                        rmsf_value = self.fallback_rmsf
                except ValueError:
                     logger.warning(f"Invalid residue ID format encountered in __getitem__: {resid_str}")
                     rmsf_value = self.fallback_rmsf


            # Convert to tensors
            voxel_tensor = torch.from_numpy(voxel_np) # Already float32
            rmsf_tensor = torch.tensor(float(rmsf_value), dtype=torch.float32)

            return voxel_tensor, rmsf_tensor

        except Exception as e:
            logger.error(f"Error retrieving prediction item {idx} ({domain_id}:{resid_str}): {str(e)}")
            # Attempt to return dummy data based on stored input_shape
            return self._get_dummy_data()

    def _get_dummy_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
         """Creates dummy data to return on error, using stored input_shape."""
         logger.debug(f"Returning dummy data with shape {self.input_shape}")
         dummy_voxel = torch.zeros(self.input_shape, dtype=torch.float32)
         dummy_rmsf = torch.tensor(0.0, dtype=torch.float32)
         return dummy_voxel, dummy_rmsf