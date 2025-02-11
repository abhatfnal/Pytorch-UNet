import h5py
import numpy as np
import json
import csv

# Load config file
config_file = "/home/abhat/wirecell_icarus/Pytorch-UNet/config-nomMC_U.json"  # Update if needed
with open(config_file, "r") as f:
    config = json.load(f)

# Read file lists
sample_list_path = config["sample_list"]
target_list_path = config["target_list"]

# Load file paths
with open(sample_list_path, "r") as f:
    sample_files = [line.strip() for line in f.readlines()]
with open(target_list_path, "r") as f:
    target_files = [line.strip() for line in f.readlines()]

# Ensure lists are of the same length
if len(sample_files) != len(target_files):
    raise ValueError("Mismatch between sample and target file lists.")

def dataset_exists(file_path, dataset_name):
    """ Check if a dataset exists in the file, handling file access errors. """
    try:
        with h5py.File(file_path, "r") as f:
            return dataset_name in f
    except (OSError, FileNotFoundError):
        print(f"Skipping {file_path} (file not found or unreadable).")
        return False

def get_nonzero_counts(file_path, dataset_name, target_key):
    """ Function to count nonzero elements for all channels, handling file access errors. """
    try:
        with h5py.File(file_path, "r") as f:
            if dataset_name in f and target_key in f[dataset_name]:
                data = np.array(f[dataset_name][target_key])  # Convert to NumPy array
                nonzero_counts = np.count_nonzero(data, axis=0)  # Count nonzero elements per channel
                return nonzero_counts
            else:
                return None
    except (OSError, FileNotFoundError):
        print(f"Skipping {file_path} (file not found or unreadable).")
        return None

# CSV file to store results
csv_filename = "nomMC_U.csv"

with open(csv_filename, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Sample File", "Target File", "Average Rec/Truth Ratio"])

    valid_ratios = []

    for sample_file, target_file in zip(sample_files, target_files):
        # Check if both files can be opened
        if not dataset_exists(sample_file, "4") or not dataset_exists(target_file, "4"):
            print(f"Skipping {sample_file} and {target_file} (dataset '4' missing or file unreadable).")
            continue

        # Get nonzero counts for all channels
        true_nonzero = get_nonzero_counts(target_file, "4", "frame_deposplat")
        rec_nonzero = get_nonzero_counts(sample_file, "4", "frame_gauss")

        if true_nonzero is not None and rec_nonzero is not None:
            valid_channels = true_nonzero > 0  # Mask for channels with at least one nonzero value in frame_deposplat
            valid_true_counts = true_nonzero[valid_channels]
            valid_rec_counts = rec_nonzero[valid_channels]

            # Compute ratios for valid channels
            ratios = np.divide(valid_rec_counts, valid_true_counts, 
                               out=np.zeros_like(valid_rec_counts, dtype=float), where=valid_true_counts > 0)

            # Compute average ratio for this file pair
            avg_ratio = np.mean(ratios)
            valid_ratios.append(avg_ratio)

            # Write result to CSV
            csv_writer.writerow([sample_file, target_file, f"{avg_ratio:.4f}"])

    # Compute overall average ratio across all valid file pairs
    if valid_ratios:
        final_avg_ratio = np.mean(valid_ratios)
        csv_writer.writerow(["Overall Average", "", f"{final_avg_ratio:.4f}"])

print(f"Results saved in {csv_filename}.")
