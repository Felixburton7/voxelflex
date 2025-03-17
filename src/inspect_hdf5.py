from voxelflex.utils.file_utils import inspect_hdf5_structure
import os

# Update this path to point to your HDF5 file
hdf5_file = os.path.expanduser("~/drFelix/data_full/processed/voxelized_output/mdcath_dataset_half_CNOCACB.hdf5")

# Inspect the file structure
inspect_hdf5_structure(hdf5_file, domain_sample=3, print_residues=True)