
rec_file = "root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/sbnd/scratch/users/gputnam/omnidetector-2/optfilter-nomMC-5/14744392_0/h5s_inspected/tpc0_plane0_rec.h5"      # Replace with the rec file path

truth_file ="root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/sbnd/scratch/users/gputnam/omnidetector-2/optfilter-nomMC-5/14744392_0/h5s_inspected/tpc0_plane0_tru.h5" 

import h5py

# Replace these with the actual paths to your files
# truth_file = "path_to_truth_file.h5"  # Replace with the truth file path
# rec_file = "path_to_rec_file.h5"      # Replace with the rec file path

def compare_frame_sizes(truth_file, rec_file, truth_tag="frame_deposplat", rec_tag="frame_gauss"):
    """
    Compare the sizes of `frame_deposplat` from the truth file and `frame_gauss` from the rec file.

    :param truth_file: Path to the truth HDF5 file.
    :param rec_file: Path to the rec HDF5 file.
    :param truth_tag: Dataset name for the truth data (default is `frame_deposplat`).
    :param rec_tag: Dataset name for the rec data (default is `frame_gauss`).
    """
    with h5py.File(truth_file, 'r') as tf, h5py.File(rec_file, 'r') as rf:
        for group in ['0', '1', '2', '3', '4']:
            truth_path = f"//{group}/{truth_tag}"  # Prefix with double slashes
            rec_path = f"//{group}/{rec_tag}"      # Prefix with double slashes

            # Check if the datasets exist
            if truth_path not in tf:
                print(f"Truth dataset not found: {truth_path}")
                continue
            if rec_path not in rf:
                print(f"Rec dataset not found: {rec_path}")
                continue

            # Load the datasets
            truth_frame = tf[truth_path]
            rec_frame = rf[rec_path]

            # Compare sizes
            print(f"Group: {group}")
            print(f"  Truth file: Dataset: {truth_path}, Shape: {truth_frame.shape}")
            print(f"  Rec file: Dataset: {rec_path}, Shape: {rec_frame.shape}")

            if truth_frame.shape == rec_frame.shape:
                print("  Sizes match!")
            else:
                print("  Sizes do not match!")

# Call the comparison function
compare_frame_sizes(truth_file, rec_file)


