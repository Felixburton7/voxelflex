# Voxelflex

A package for predicting per-residue RMSF values from voxelized protein data.

## Overview

Voxelflex is a Python package that uses 3D convolutional neural networks to predict per-residue RMSF (Root Mean Square Fluctuation) values from voxelized protein structures. The package provides a complete machine learning pipeline including data loading, model training, prediction, evaluation, and visualization.

## Features

- Load and validate voxelized protein data (.hdf5 files)
- Load and validate RMSF data (.csv files)
- Train CNN models with different architectures
- Make RMSF predictions on new protein structures
- Evaluate model performance with various metrics
- Generate visualizations for analysis
- Automatically detect and utilize available GPU resources
- Optimize processing based on available CPU cores
- Provide detailed logging with progress tracking

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/username/voxelflex.git
cd voxelflex

# Install the package in development mode
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

## Usage

### Configuration

Voxelflex uses YAML configuration files to specify parameters for the pipeline. Create a YAML file with the following sections:

- `input`: Specify paths to input data (voxel and RMSF files)
- `output`: Specify output directory for logs, models, metrics, and visualizations
- `model`: Configure model architecture and parameters
- `training`: Configure training parameters
- `logging`: Configure logging behavior
- `visualization`: Configure visualization options
- `system_utilization`: Configure system resource utilization

Example configuration file:

```yaml
input:
  data_dir: ~/data_full
  voxel_file: ~/data_full/voxel_data.hdf5
  rmsf_dir: ~/data_full/processed/RMSF/replicas/replica_average
  temperature: 320
  domain_ids: []
  use_metadata: true

output:
  base_dir: outputs/
  log_file: voxelflex.log

model:
  architecture: dilated_resnet3d
  input_channels: 5
  channel_growth_rate: 1.5
  num_residual_blocks: 4
  dropout_rate: 0.3
  base_filters: 32

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 1e-5
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  seed: 42

logging:
  level: INFO
  console_level: INFO
  file_level: DEBUG
  show_progress_bars: true

visualization:
  plot_loss: true
  plot_predictions: true
  plot_residue_type_analysis: true
  plot_error_distribution: true
  plot_amino_acid_performance: true
  save_format: png
  dpi: 300
  plot_correlation: true
  max_scatter_points: 1000

system_utilization:
  detect_cores: true
  adjust_for_gpu: true
  num_workers: 4
```

### Running the Pipeline

To run the complete pipeline (train, predict, evaluate, visualize) with a single command:

```bash
voxelflex run --config path/to/config.yaml
```

This will:
1. Train a model using the specified architecture
2. Make predictions on the test set
3. Evaluate model performance using various metrics
4. Generate visualizations of the results

### Individual Commands

You can also run individual steps of the pipeline:

```bash
# Train a model
voxelflex train --config path/to/config.yaml

# Make predictions with a trained model
voxelflex predict --config path/to/config.yaml --model path/to/model.pt

# Evaluate model performance
voxelflex evaluate --config path/to/config.yaml --model path/to/model.pt

# Create visualizations
voxelflex visualize --config path/to/config.yaml --predictions path/to/predictions.csv
```

## Output Structure

The pipeline creates the following output structure:

```
outputs/
├── logs/
│   └── voxelflex.log
├── models/
│   ├── model_architecture_timestamp.pt
│   └── training_history_timestamp.json
├── metrics/
│   ├── predictions_timestamp.csv
│   └── metrics_timestamp.json
└── visualizations/
    ├── loss_curve_timestamp.png
    ├── prediction_scatter_timestamp.png
    ├── error_distribution_timestamp.png
    ├── residue_type_analysis_timestamp.png
    └── amino_acid_performance_timestamp.png
```

## Model Architectures

Voxelflex includes several CNN architectures for RMSF prediction:

- `voxelflex_cnn`: Basic 3D CNN with multiple convolutional layers
- `dilated_resnet3d`: ResNet-style architecture with dilated convolutions
- `multipath_rmsf_net`: Multi-path network for capturing features at different scales

## License

MIT License


## MORE 

How the Components Work Together
Now that we've implemented all the necessary components, let's explain how they work together to create a complete ML pipeline:

Command-Line Interface (cli.py): Provides a user-friendly interface to run the pipeline. It parses command-line arguments and dispatches the appropriate commands.
Configuration Module (config.py): Loads and validates configuration from YAML files, ensuring all required parameters are present.
Data Loading Module (data_loader.py): Loads voxel data from HDF5 files and RMSF data from CSV files. It uses the validators.py module to ensure data quality.
Model Architecture (cnn_models.py): Defines several CNN architectures for RMSF prediction, each with different characteristics.
Training Command (train.py): Trains a model using the specified architecture, data, and training parameters. It saves the trained model and training history.
Prediction Command (predict.py): Makes predictions using a trained model and saves the results.
Evaluation Command (evaluate.py): Calculates performance metrics such as MSE, RMSE, MAE, and R² for a trained model.
Visualization Command (visualize.py): Creates visualizations of model performance, including loss curves, prediction scatter plots, error distributions, and residue type analysis.
Utility Modules:

file_utils.py: Handles file and directory operations.
logging_utils.py: Provides custom logging functionality with a fixed-bottom progress bar.
system_utils.py: Detects and utilizes system resources (CPU cores, GPU availability).


Run Pipeline Function (run_pipeline in cli.py): Executes all steps in sequence (train, predict, evaluate, visualize) using a single configuration file.

How to Install and Use the Package
To install and use the Voxelflex package:

Installation:
bashCopy# Clone or download the repository
cd voxelflex

# Install the package
pip install -e .

Create a Configuration File:
Create a YAML configuration file with the necessary parameters (see the README for details).
Run the Pipeline:
bashCopyvoxelflex run --config path/to/config.yaml

Check Output:
The pipeline creates an outputs directory with logs, trained models, metrics, and visualizations.

Summary
The Voxelflex package provides a comprehensive solution for predicting per-residue RMSF values from voxelized protein data. It integrates data loading, model training, prediction, evaluation, and visualization into a unified workflow. The package also includes system resource detection, GPU support, and custom logging with a progress bar.
Key benefits of this design:

Modular Structure: Each component can be used independently or as part of the pipeline.
Configurability: All parameters can be adjusted through a single YAML configuration file.
Resource Optimization: Automatically detects and utilizes available system resources.
Comprehensive Logging: Provides detailed logs with progress tracking.
Visual Analysis: Generates visualizations for model performance analysis.

# voxelflex
