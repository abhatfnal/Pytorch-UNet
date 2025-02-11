# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# import torch
# import os

# # Import necessary functions from your script
# from utils import hwc_to_chw, h5_utils as h5u

# # Define file paths
# # true_mask_file = "root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/sbnd/scratch/users/gputnam/omnidetector-2/nugen/optfilter-numi-nomMC-5/84373750_9/h5s_inspected/tpc3_plane0_tru.h5"
# # rec_mask_file = "root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/sbnd/scratch/users/gputnam/omnidetector-2/nugen/optfilter-numi-nomMC-5/84373750_9/h5s_inspected/tpc3_plane0_rec.h5"

# true_mask_file = "/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/samples/opaqueMC/h5s_inspected/tpc0_plane0_tru.h5"
# rec_mask_file = "/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/samples/opaqueMC/h5s_inspected/tpc0_plane0_rec.h5"

# # Load DNN model
# model_u_path = '/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/training/mixed_samples/nom/U_Plane/best_loss.pth'
# net_u = torch.jit.load(model_u_path, map_location='cpu')
# net_u.eval()

# def get_waveform(file_path, dataset_name, key, channel_num):
#     """ Extracts waveform for a given key and channel. """
#     with h5py.File(file_path, "r") as f:
#         if dataset_name in f and key in f[dataset_name]:
#             data = np.array(f[dataset_name][key])
#             return data[:, channel_num]
#         else:
#             print(f"{key} not found in dataset '{dataset_name}' in {file_path}")
#             return None

# def predict_img(net, full_img):
#     img_tensor = torch.from_numpy(hwc_to_chw(full_img)).unsqueeze(0).float()
#     with torch.no_grad():
#         full_mask = net(img_tensor).squeeze().cpu().numpy()
#     return full_mask

# def plot_waveforms(channel_num, true_waveform, rec_waveform, dnn_waveform):
#     """ Plots the waveforms for comparison with dual y-axes. """
#     fig, ax1 = plt.subplots(figsize=(10, 6))
    
#     # Corrected time axis mappings
#     time_ticks = np.linspace(0, 4096, len(true_waveform))  # True waveform time axis
#     dnn_time_ticks = np.linspace(0, 4096, dnn_waveform.shape[1])  # DNN time axis mapped to match true waveform scale
    
#     # Primary y-axis for frame_deposplat and frame_gauss
#     ax1.set_xlabel("Time Tick")
#     ax1.set_ylabel("Amplitude", color='black')
#     ax1.plot(time_ticks, true_waveform, label="True Mask - frame_deposplat", lw=1, color='black')
#     ax1.plot(time_ticks, rec_waveform, label="Reconstructed Mask - frame_gauss", lw=1, color='red', linestyle='dashed')
#     ax1.tick_params(axis='y', labelcolor='black')
#     # ax1.set_xlim(0, 2200)
    
#     # Secondary y-axis for DNN prediction
#     ax2 = ax1.twinx()
#     ax2.set_ylabel("DNN Prediction", color='green')
#     ax2.plot(dnn_time_ticks, dnn_waveform[channel_num, :], label="DNN Prediction", lw=1, color='green', linestyle='dotted')
#     ax2.tick_params(axis='y', labelcolor='green')
    
#     # Legends
#     ax1.legend(loc='upper left')
#     ax2.legend(loc='upper right')
    
#     plt.title(f"Waveform Comparison for Channel {channel_num}")
#     plt.grid(True)
    
#     plt.savefig('waveform_comparison.pdf')
#     plt.show()

# # Extract waveforms
# channel_num = 100
# dataset_name = "4"
# true_waveform = get_waveform(true_mask_file, dataset_name, "frame_deposplat", channel_num)
# rec_waveform = get_waveform(rec_mask_file, dataset_name, "frame_gauss", channel_num)
# dnn_input = h5u.get_hwc_img(rec_mask_file, dataset_name, ["frame_looseLf", "frame_mp3ROI"], [1, 8], [0, 2112], [0, 4096], 2000)

# dnn_waveform = predict_img(net_u, dnn_input) if dnn_input is not None else None

# # Plot waveforms
# plot_waveforms(channel_num, true_waveform, rec_waveform, dnn_waveform)


import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import os

# Import necessary functions from your script
from utils import hwc_to_chw, h5_utils as h5u

# Set plane number: 0 for plane0 and 1 for plane1
plane = 1  # Change to 1 for plane1 files

if plane == 0:
    true_mask_file = "/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/samples/opaqueMC/h5s_inspected/tpc0_plane0_tru.h5"
    rec_mask_file  = "/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/samples/opaqueMC/h5s_inspected/tpc0_plane0_rec.h5"
    model_path     = "/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/training/mixed_samples/nom/U_Plane/best_loss.pth"
    channel_range  = [0, 2112]
elif plane == 1:
    true_mask_file = "/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/samples/opaqueMC/h5s_inspected/tpc0_plane1_tru.h5"
    rec_mask_file  = "/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/samples/opaqueMC/h5s_inspected/tpc0_plane1_rec.h5"
    model_path     = "/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/training/mixed_samples/nom/V_Plane/best_loss.pth"
    channel_range  = [0, 5600]
else:
    raise ValueError("Invalid plane number. Please set plane to 0 or 1.")

# Load DNN model
net = torch.jit.load(model_path, map_location='cpu')
net.eval()

def get_waveform(file_path, dataset_name, key, channel_num):
    """Extracts waveform for a given key and channel."""
    with h5py.File(file_path, "r") as f:
        if dataset_name in f and key in f[dataset_name]:
            data = np.array(f[dataset_name][key])
            return data[:, channel_num]
        else:
            print(f"{key} not found in dataset '{dataset_name}' in {file_path}")
            return None

def predict_img(net, full_img):
    img_tensor = torch.from_numpy(hwc_to_chw(full_img)).unsqueeze(0).float()
    with torch.no_grad():
        full_mask = net(img_tensor).squeeze().cpu().numpy()
    return full_mask

def plot_waveforms(channel_num, true_waveform, rec_waveform, dnn_waveform):
    """Creates a waveform comparison plot for a given channel and returns the figure."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Define time axes (adjust as needed)
    time_ticks = np.linspace(0, 4096, len(true_waveform))
    dnn_time_ticks = np.linspace(0, 4096, dnn_waveform.shape[1])
    
    # Plot true and reconstructed waveforms on primary y-axis
    ax1.set_xlabel("Time Tick")
    ax1.set_ylabel("Amplitude", color='black')
    ax1.plot(time_ticks, true_waveform, label="True Mask - frame_deposplat", lw=1, color='black')
    ax1.plot(time_ticks, rec_waveform, label="Reconstructed Mask - frame_gauss", lw=1, color='red', linestyle='dashed')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Plot DNN prediction on secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("DNN Prediction", color='green')
    ax2.plot(dnn_time_ticks, dnn_waveform[channel_num, :], label="DNN Prediction", lw=1, color='green', linestyle='dotted')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Title and legends
    plt.title(f"Waveform Comparison for Channel {channel_num}")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True)
    
    return fig

# List of channel numbers for which to generate waveforms
# channel_nums = [10, 100, 500]  # Modify this list as needed
channel_nums=np.arange(2400,2500,1)
dataset_name = "4"
# Compute the DNN input once based on the channel range.
dnn_input = h5u.get_hwc_img(rec_mask_file, dataset_name, 
                            ["frame_looseLf", "frame_mp3ROI"], 
                            [1, 8], 
                            channel_range, 
                            [0, 4096], 
                            2000)
dnn_waveform = predict_img(net, dnn_input) if dnn_input is not None else None

# Create a PDF file and add each channel's plot as a separate page.
pdf_filename = "waveform_comparison.pdf"
with PdfPages(pdf_filename) as pdf:
    for ch in channel_nums:
        true_waveform = get_waveform(true_mask_file, dataset_name, "frame_deposplat", ch)
        rec_waveform = get_waveform(rec_mask_file, dataset_name, "frame_gauss", ch)
        if true_waveform is None or rec_waveform is None or dnn_waveform is None:
            print(f"Skipping channel {ch} due to missing data.")
            continue
        fig = plot_waveforms(ch, true_waveform, rec_waveform, dnn_waveform)
        pdf.savefig(fig)
        plt.close(fig)

print(f"Waveform comparison PDF saved as {pdf_filename}")

