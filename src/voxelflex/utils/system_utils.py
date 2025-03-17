"""
System utility functions for Voxelflex.

This module provides functions for detecting and utilizing system resources.
"""

import os
import platform
import multiprocessing
from typing import Dict, Any, Tuple

import torch

from voxelflex.utils.logging_utils import get_logger

logger = get_logger(__name__)

def get_device(adjust_for_gpu: bool = True) -> torch.device:
    """
    Get appropriate device (CPU or GPU) for PyTorch.
    
    Args:
        adjust_for_gpu: Whether to use GPU if available
        
    Returns:
        PyTorch device
    """
    if adjust_for_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def check_system_resources(detect_cores: bool = True, 
                           adjust_for_gpu: bool = True) -> Dict[str, Any]:
    """
    Check available system resources.
    
    Args:
        detect_cores: Whether to detect available CPU cores
        adjust_for_gpu: Whether to check for GPU availability
        
    Returns:
        Dictionary containing system resource information
    """
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    
    # Check CPU resources
    if detect_cores:
        cpu_count = multiprocessing.cpu_count()
        system_info["cpu_count"] = cpu_count
        system_info["recommended_workers"] = max(1, cpu_count // 2)
    
    # Check GPU resources
    if adjust_for_gpu:
        system_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            system_info["cuda_device_count"] = torch.cuda.device_count()
            system_info["cuda_device_name"] = torch.cuda.get_device_name(0)
            system_info["cuda_version"] = torch.version.cuda
    
    # Log system information
    logger.info(f"Platform: {system_info['platform']}")
    logger.info(f"Python version: {system_info['python_version']}")
    logger.info(f"PyTorch version: {system_info['torch_version']}")
    
    if detect_cores:
        logger.info(f"CPU cores: {system_info['cpu_count']}")
        logger.info(f"Recommended worker processes: {system_info['recommended_workers']}")
    
    if adjust_for_gpu and system_info.get("cuda_available", False):
        logger.info(f"CUDA available: {system_info['cuda_available']}")
        logger.info(f"CUDA device count: {system_info['cuda_device_count']}")
        logger.info(f"CUDA device name: {system_info['cuda_device_name']}")
        logger.info(f"CUDA version: {system_info['cuda_version']}")
    
    return system_info


def set_num_threads(num_threads: int = None) -> int:
    """
    Set number of threads for PyTorch.
    
    Args:
        num_threads: Number of threads to use (if None, use all available cores)
        
    Returns:
        Number of threads set
    """
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()
    
    torch.set_num_threads(num_threads)
    logger.info(f"Set PyTorch number of threads to {num_threads}")
    
    return num_threads