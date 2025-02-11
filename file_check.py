# import h5py

# # Replace 'your_file.h5' with the path to your HDF5 file
# file_path = 'root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/sbnd/scratch/users/gputnam/omnidetector-2/nugen/optfilter-bnb-nomMC-5/15408720_0/h5s_inspected/tpc0_plane0_rec.h5'

# # file_path = 'root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/sbnd/scratch/users/gputnam/omnidetector-2/optfilter-nomMC-5/83842458_9/h5s_inspected/tpc3_plane0_tru.h5'

# # file_path ='/pnfs/sbnd/scratch/users/gputnam/omnidetector-2/nugen/optfilter-numi-nomMC-5/84373750_9/h5s_inspected/tpc3_plane1_tru.h5'

# def explore_h5_file(file_path):
#     """
#     Explore the contents of an HDF5 file and print its structure.

#     Args:
#         file_path (str): Path to the HDF5 file.
#     """
#     with h5py.File(file_path, 'r') as h5file:
#         print("Contents of the HDF5 file:")
        
#         def print_structure(name, obj):
#             """Helper function to print the structure of the HDF5 file."""
#             if isinstance(obj, h5py.Dataset):
#                 print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
#             elif isinstance(obj, h5py.Group):
#                 print(f"Group: {name}")
        
#         h5file.visititems(print_structure)

# # Call the function to explore the file
# explore_h5_file(file_path)

import h5py
from tqdm import tqdm

def count_groups(file_list_path, output_file_path):
    # Initialize a dictionary to store file paths for specific group counts
    group_files = {i: [] for i in range(6)}  # Files with 0 to 5 groups

    # Read the file containing the list of HDF5 file paths
    with open(file_list_path, 'r') as file:
        file_paths = [line.strip() for line in file if line.strip()]

    # Use tqdm to display the progress bar
    for h5_path in tqdm(file_paths, desc="Processing files"):
        try:
            with h5py.File(h5_path, 'r') as h5file:
                # Count the number of groups in the HDF5 file
                num_groups = len([key for key in h5file.keys() if isinstance(h5file[key], h5py.Group)])
                # Add the file path to the corresponding group count list
                if num_groups in group_files:
                    group_files[num_groups].append(h5_path)
                else:
                    print(f"File {h5_path} has more than 5 groups ({num_groups}). Ignoring.")
        except Exception as e:
            print(f"Error processing file {h5_path}: {e}")

    # Save the results to the output file
    with open(output_file_path, 'w') as output_file:
        for num_groups, files in group_files.items():
            output_file.write(f"Files with {num_groups} groups:\n")
            for file_path in files:
                output_file.write(f"{file_path}\n")
            output_file.write("\n")  # Add a blank line for better readability

    print(f"Results have been saved to {output_file_path}")

# Example usage
file_list_path = "/exp/sbnd/data/users/abhat/wirecell_data/dnn_roi/ICARUS/samples/target_nom_U.list"  # Replace with your file list path
output_file_path = "group_counts_output.file"  # Replace with your desired output file path
count_groups(file_list_path, output_file_path)



