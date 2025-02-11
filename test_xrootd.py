# import h5py
# import os

# # If you need to preload XRootD for remote access, run this script with:
# #   LD_PRELOAD=/lib64/libXrdPosixPreload.so:$LD_PRELOAD python verify_datasets_rec_vs_tru.py

# FOLDER_PATH = "root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/sbnd/scratch/users/gputnam/omnidetector-3/nugen/optfilter-bnb-nomMC-5/59993016_10/h5s_inspected/"

# # Datasets only expected in reconstruction files (rec)
# REQUIRED_DATASETS_REC = [
#     "frame_looseLf",
#     "frame_mp2ROI",
#     "frame_mp3ROI",
#     "frame_gauss"
# ]

# # Datasets only expected in truth files (tru)
# REQUIRED_DATASETS_TRU = [
#     "frame_deposplat"
# ]

# def verify_datasets_in_file(file_path, datasets_to_check):
#     """
#     Opens the HDF5 file and checks each numbered group (0, 1, 2, ...) 
#     for all required datasets in 'datasets_to_check'.
#     Prints a warning if any required dataset is missing in a group.
#     """
#     print(f"\nVerifying file: {file_path}")
#     try:
#         with h5py.File(file_path, 'r') as f:
#             # Sort groups by numeric order if they are purely digits
#             groups = sorted(
#                 f.keys(),
#                 key=lambda x: int(x) if x.isdigit() else float('inf')
#             )

#             if not groups:
#                 print(f"  No groups found in {file_path}")
#                 return

#             # Check each group for required datasets
#             for group in groups:
#                 for dataset in datasets_to_check:
#                     full_path = f"{group}/{dataset}"
#                     if full_path not in f:
#                         print(f"  WARNING: Missing '{dataset}' in group '{group}' of file '{file_path}'")

#     except OSError as e:
#         print(f"  ERROR opening file {file_path}: {e}")

# def verify_file(file_path):
#     """
#     Determines whether file_path is tru or rec, 
#     then verifies the appropriate datasets.
#     """
#     if file_path.endswith("_rec.h5"):
#         # rec file
#         verify_datasets_in_file(file_path, REQUIRED_DATASETS_REC)
#     elif file_path.endswith("_tru.h5"):
#         # tru file
#         verify_datasets_in_file(file_path, REQUIRED_DATASETS_TRU)
#     else:
#         print(f"\nSkipping {file_path}: not recognized as tru or rec file.")

# def verify_all_files_in_folder(folder_path):
#     """
#     Verifies all tpc-plane combos in the provided folder 
#     and checks them against the appropriate dataset list.
#     """
#     # We'll just systematically check tpc=0..3, plane=0..1:
#     combos = []
#     for tpc in [0,1,2,3]:
#         for plane in [0,1]:
#             combos.append(f"tpc{tpc}_plane{plane}_tru.h5")
#             combos.append(f"tpc{tpc}_plane{plane}_rec.h5")

#     for fname in combos:
#         file_path = os.path.join(folder_path, fname)
#         verify_file(file_path)

# if __name__ == "__main__":
#     verify_all_files_in_folder(FOLDER_PATH)
#     print("\nDataset verification complete.")

import h5py

def print_structure(name, obj):
    # Create an indent based on the depth in the file structure
    indent = '  ' * (name.count('/') - 1)
    if isinstance(obj, h5py.Group):
        print(f"{indent}Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")

# Replace 'your_file.h5' with the path to your h5 file.
with h5py.File('downsampled_all_events.h5', 'r') as hdf:
    # Recursively visit all objects in the file and print their structure.
    hdf.visititems(print_structure)

