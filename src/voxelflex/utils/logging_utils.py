"""
Logging utilities for Voxelflex.

This module provides custom logging functionality, including a fixed-bottom progress bar.
"""

import os
import sys
import time
import logging
from typing import Optional, Dict, Any
import shutil

# ANSI escape codes for colors
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_MAGENTA = "\033[35m"
COLOR_CYAN = "\033[36m"

# ASCII art for logger
ASCII_HEADER = r"""
 __      __                  _  __ _           
 \ \    / /                 | |/ _| |          
  \ \  / /__  __  _____  ___| | |_| | _____  __
   \ \/ / _ \\ \/ / _ \/ __| |  _| |/ _ \ \/ /
    \  / (_) |>  <  __/ (__| | | | |  __/>  < 
     \/ \___//_/\_\___|\___|_|_| |_|\___/_/\_\                                     
"""

def setup_logging(log_file: Optional[str] = None, console_level: str = "INFO", 
                  file_level: str = "DEBUG") -> None:
    """
    Set up logging for Voxelflex.
    
    Args:
        log_file: Path to log file (if None, logging to file is disabled)
        console_level: Logging level for console output
        file_level: Logging level for file output
    """
    # Convert string levels to logging levels
    console_level = getattr(logging, console_level.upper())
    file_level = getattr(logging, file_level.upper())
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture everything
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter with timestamp
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Log ASCII art header
    logger.info(f"\n{ASCII_HEADER}")
    logger.info("Voxelflex logger initialized")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ProgressBar:
    """
    Fixed-bottom progress bar that can be updated from anywhere in the code.
    """
    
    def __init__(self, total: int, prefix: str = "", suffix: str = "", 
                 bar_length: int = 30):
        """
        Initialize progress bar.
        
        Args:
            total: Total number of items
            prefix: Prefix string
            suffix: Suffix string
            bar_length: Length of the progress bar
        """
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.bar_length = bar_length
        self.current = 0
        self.start_time = time.time()
        self.last_printed_length = 0
        
        # Get terminal width
        self.terminal_width = shutil.get_terminal_size().columns
    
    def update(self, current: int) -> None:
        """
        Update progress bar.
        
        Args:
            current: Current progress value
        """
        self.current = current
        self._print_progress()
    
    def _print_progress(self) -> None:
        """Print the progress bar."""
        # Calculate progress
        percent = float(self.current) / float(self.total) * 100
        filled_length = int(round(self.bar_length * self.current / float(self.total)))
        bar = "█" * filled_length + "-" * (self.bar_length - filled_length)
        
        # Calculate elapsed time and ETA
        elapsed_time = time.time() - self.start_time
        items_per_second = self.current / elapsed_time if elapsed_time > 0 else 0
        eta = (self.total - self.current) / items_per_second if items_per_second > 0 else 0
        
        # Format time strings
        elapsed_str = self._format_time(elapsed_time)
        eta_str = self._format_time(eta)
        
        # Build progress string
        progress_str = f"\r{self.prefix} [{bar}] {percent:.1f}% {self.current}/{self.total} "
        progress_str += f"[{elapsed_str}<{eta_str}, {items_per_second:.1f} it/s] {self.suffix}"
        
        # Ensure the string fits the terminal width
        if len(progress_str) > self.terminal_width:
            progress_str = progress_str[:self.terminal_width - 3] + "..."
        
        # Clear previous output by printing spaces
        clear_str = " " * self.last_printed_length
        sys.stdout.write("\r" + clear_str)
        
        # Print progress
        sys.stdout.write(progress_str)
        sys.stdout.flush()
        
        self.last_printed_length = len(progress_str)
    
    def _format_time(self, seconds: float) -> str:
        """
        Format time in seconds to a readable string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            seconds = seconds % 60
            return f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(seconds / 3600)
            seconds = seconds % 3600
            minutes = int(seconds / 60)
            seconds = seconds % 60
            return f"{hours}h {minutes}m {seconds:.1f}s"
    
    def finish(self) -> None:
        """Complete the progress bar and move to the next line."""
        self.update(self.total)
        sys.stdout.write("\n")
        sys.stdout.flush()