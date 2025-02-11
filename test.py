import h5py
import numpy as np
import matplotlib.pyplot as plt

# Define file paths
true_mask_file = "root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/sbnd/scratch/users/gputnam/omnidetector-2/nugen/optfilter-numi-nomMC-5/84373750_9/h5s_inspected/tpc3_plane0_tru.h5"
rec_mask_file = "root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/sbnd/scratch/users/gputnam/omnidetector-2/nugen/optfilter-numi-nomMC-5/84373750_9/h5s_inspected/tpc3_plane0_rec.h5"

def get_waveform(file_path, dataset_name, target_key, channel_num):
    """ Function to extract the waveform for a given channel """
    with h5py.File(file_path, "r") as f:
        if dataset_name in f and target_key in f[dataset_name]:
            data = np.array(f[dataset_name][target_key])  # Convert to NumPy array
            return data[:, channel_num]  # All time ticks for the given channel
        else:
            print(f"{target_key} not found in dataset '{dataset_name}' in {file_path} file.")
            return None

# Channel to plot
channel_num = 100

# Extract waveforms
true_waveform = get_waveform(true_mask_file, "4", "frame_deposplat", channel_num)
rec_waveform = get_waveform(rec_mask_file, "4", "frame_gauss", channel_num)

# Plot both waveforms on the same figure
plt.figure(figsize=(10, 5))
if true_waveform is not None:
    plt.plot(true_waveform, label=f"True Mask - frame_deposplat (Channel {channel_num})", lw=1, color='blue')
if rec_waveform is not None:
    plt.plot(rec_waveform, label=f"Reconstructed Mask - frame_gauss (Channel {channel_num})", lw=1, color='red', linestyle='dashed')

plt.xlabel("Time Tick (0 to 4095)")
plt.ylabel("Amplitude")
plt.title(f"Waveform Comparison for Channel {channel_num}")
plt.legend()
plt.grid(True)
plt.savefig('waveform_comparison.pdf')
plt.show()
