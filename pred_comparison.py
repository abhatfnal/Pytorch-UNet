# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# import torch
# import os

# # Import necessary functions from your script
# from utils import hwc_to_chw, h5_utils as h5u

# # Main folder containing HDF5 files
# main_folder = '/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/samples/opaqueMC/'

# # main_folder ='root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr/sbnd/scratch/users/gputnam/omnidetector-2/nugen/optfilter-numi-nomMC-5/84373750_9/h5s_inspected/'

# # TPCs and plane mappings
# tpcs = [0, 1, 2, 3]
# # tpcs=[3]
# plane_suffix_map = {'front_induction': '0', 'middle_induction': '1'}
# event_to_plot = 4  # The event index to plot

# # Load U and V Plane models
# model_u_path = '/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/training/mixed_samples/nom/U_Plane/best_loss.pth'
# model_v_path = '/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/training/mixed_samples/nom/V_Plane/best_loss.pth'

# net_u = torch.jit.load(model_u_path, map_location='cpu')
# net_u.eval()
# net_v = torch.jit.load(model_v_path, map_location='cpu')
# net_v.eval()

# def predict_img(net, full_img, scale_factor=0.5, out_threshold=0.5, use_gpu=False):
#     img_tensor = torch.from_numpy(hwc_to_chw(full_img))
#     if use_gpu:
#         img_tensor = img_tensor.cuda()

#     with torch.no_grad():
#         input = img_tensor.unsqueeze(0)
#         full_mask = net(input).squeeze().cpu().numpy()

#     if out_threshold < 0:
#         return full_mask

#     return full_mask > out_threshold

# def model_selection(file_path, plane):
#     if plane == "U":
#         net = net_u
#         img = h5u.get_hwc_img(file_path, event_to_plot, ['frame_looseLf', 'frame_mp3ROI'], [1, 8], [0, 2112], [0, 4096], 2000)
#     elif plane == "V":
#         net = net_v
#         img = h5u.get_hwc_img(file_path, event_to_plot, ['frame_looseLf', 'frame_mp3ROI'], [1, 8], [0, 5600], [0, 4096], 2000)

#     if img is None:
#         print(f"No image data found for {plane} plane in file {file_path}")
#         return None

#     predicted_mask = predict_img(net=net, full_img=img, scale_factor=0.5, out_threshold=0.5, use_gpu=False)
#     print(f"Predicted image shape for {plane} plane: {predicted_mask.shape}")
#     return np.transpose(predicted_mask)

# def plot_data(tpc, plane, plane_suffix, data_dict, event_idx, predicted_image, pdf):
#     fig, axes = plt.subplots(3, 2, figsize=(15, 15))
#     titles = ["DepoSplat", "LooseLF", "MP3ROI", "MP2ROI", "Gauss", "Predicted"]

#     for i, (key, data) in enumerate(data_dict.items()):
#         if data is not None and np.any(data):
#             print(f"Plotting {key} for TPC {tpc} Plane {plane_suffix} Event {event_idx}, shape: {data.shape}")
#             ax = axes[i//2, i%2]
#             ax.imshow(data, aspect='auto', cmap="bwr", vmin=-1000, vmax=1000, origin='lower')
#             ax.set_title(f"{titles[i]} TPC{tpc} Plane{plane_suffix} (Event {event_idx})")
#             ax.set_xlabel("Channels")
#             ax.set_ylabel("Ticks")

#     # Predicted image plot
#     if predicted_image is not None and np.any(predicted_image):
#         print(f"Plotting Predicted Image for TPC {tpc} Plane {plane_suffix} Event {event_idx}, shape: {predicted_image.shape}")
#         ax = axes[2, 1]
#         ax.imshow(predicted_image, aspect='auto', cmap="bwr", origin='lower')
#         ax.set_title(f"Predicted Image TPC{tpc} Plane{plane_suffix} (Event {event_idx})")
#         ax.set_xlabel("Channels")
#         ax.set_ylabel("Ticks")
#     else:
#         print(f"Predicted image is empty or None for TPC {tpc} Plane {plane_suffix}")

#     plt.tight_layout()
#     pdf.savefig(fig)
#     plt.close()

# # Iterate through all subdirectories in the main folder
# with PdfPages("ICARUS_images_model_nomMC_predict_opaqueMC.pdf") as pdf:
#     print("Starting data loading and plotting...")
    
#     for folder_name in os.listdir(main_folder):
#         folder_path = os.path.join(main_folder, folder_name)
#         print(folder_path)
        
#         if os.path.isdir(folder_path):
#             for tpc in tpcs:
#                 for plane, plane_suffix in plane_suffix_map.items():
#                     true_file_path = os.path.join(folder_path, f"tpc{tpc}_plane{plane_suffix}_tru.h5")
#                     rec_file_path = os.path.join(folder_path, f"tpc{tpc}_plane{plane_suffix}_rec.h5")
#                     print(true_file_path)
#                     print(rec_file_path)
                    

#                     if os.path.exists(true_file_path) and os.path.exists(rec_file_path):
#                         print(f"\nProcessing files:\n - True: {true_file_path}\n - Rec: {rec_file_path}")
                        
#                         with h5py.File(true_file_path, 'r') as true_file, h5py.File(rec_file_path, 'r') as rec_file:
#                             events = list(true_file.keys())
#                             if 0 <= event_to_plot < len(events):
#                                 event = events[event_to_plot]
#                                 print(f"Loading event: {event}")

#                                 true_data = true_file[event]['frame_deposplat'][:]
#                                 rec_data_looseLf = rec_file[event]['frame_looseLf'][:]
#                                 rec_data_mp3 = rec_file[event]['frame_mp3ROI'][:]
#                                 rec_data_mp2 = rec_file[event]['frame_mp2ROI'][:]
#                                 rec_data_gauss = rec_file[event]['frame_gauss'][:]

#                                 data_dict = {
#                                     "DepoSplat": true_data,
#                                     "LooseLF": rec_data_looseLf,
#                                     "MP3ROI": rec_data_mp3,
#                                     "MP2ROI": rec_data_mp2,
#                                     "Gauss": rec_data_gauss
#                                 }

#                                 predicted_image = model_selection(rec_file_path, "U" if plane_suffix == "0" else "V")

#                                 # Call the plotting function
#                                 plot_data(tpc, plane, plane_suffix, data_dict, event_to_plot, predicted_image, pdf)
#                             else:
#                                 print(f"Event index {event_to_plot} is out of range for TPC {tpc}, Plane {plane}.")
#                     else:
#                         print(f"Files for TPC {tpc}, Plane {plane} not found in folder {folder_name}.")

# print("All images have been saved to a single PDF.")
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import os

# Import necessary functions from your script
from utils import hwc_to_chw, h5_utils as h5u

# Main folder containing HDF5 files
main_folder = '/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/samples/opaqueMC/'

# TPCs and plane mappings
tpcs = [0, 1, 2, 3]
tpcs=[0]
plane_suffix_map = {'front_induction': '0', 'middle_induction': '1'}
event_to_plot = 4  # The event index to plot

# Load U and V Plane models
model_u_path = '/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/training/mixed_samples/nom/U_Plane/best_loss.pth'
model_v_path = '/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/training/mixed_samples/nom/V_Plane/best_loss.pth'

net_u = torch.jit.load(model_u_path, map_location='cpu')
net_u.eval()
net_v = torch.jit.load(model_v_path, map_location='cpu')
net_v.eval()

def predict_img(net, full_img, scale_factor=0.5, out_threshold=0.5, use_gpu=False):
    """
    Generate a predicted mask from the full image using the given network.
    """
    img_tensor = torch.from_numpy(hwc_to_chw(full_img))
    if use_gpu:
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        input_tensor = img_tensor.unsqueeze(0)
        full_mask = net(input_tensor).squeeze().cpu().numpy()

    # If out_threshold is negative, return the raw output.
    if out_threshold < 0:
        return full_mask

    # Otherwise, return a boolean mask.
    return full_mask > out_threshold

def model_selection(file_path, plane):
    """
    Select the appropriate model and image based on the plane.
    """
    if plane == "U":
        net = net_u
        img = h5u.get_hwc_img(file_path, event_to_plot, 
                              ['frame_looseLf', 'frame_mp3ROI'], 
                              [1, 8], [0, 2112], [0, 4096], 2000)
    elif plane == "V":
        net = net_v
        img = h5u.get_hwc_img(file_path, event_to_plot, 
                              ['frame_looseLf', 'frame_mp3ROI'], 
                              [1, 8], [0, 5600], [0, 4096], 2000)

    if img is None:
        print(f"No image data found for {plane} plane in file {file_path}")
        return None

    # Get the predicted mask (boolean values).
    predicted_mask = predict_img(net=net, full_img=img, scale_factor=0.5, out_threshold=0.5, use_gpu=False)
    print(f"Predicted image shape for {plane} plane: {predicted_mask.shape}")

    # --- Modification ---
    # Instead of mapping from 0->1 to -1000->1000 (which makes background blue),
    # we map 0->1 to 0->1000. That is, False (background) becomes 0 (white)
    # and True (ROI) becomes 1000 (red), which is directly comparable to gauss.
    predicted_mask = predicted_mask.astype(np.float32) * 1000
    # ---------------------

    # Transpose if needed to match orientation of the other plots.
    return np.transpose(predicted_mask)

def plot_data(tpc, plane, plane_suffix, data_dict, event_idx, predicted_image, pdf):
    """
    Plot the true and reconstructed data alongside the predicted image.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    titles = ["DepoSplat", "LooseLF", "MP3ROI", "MP2ROI", "Gauss", "Predicted"]

    for i, (key, data) in enumerate(data_dict.items()):
        if data is not None and np.any(data):
            print(f"Plotting {key} for TPC {tpc} Plane {plane_suffix} Event {event_idx}, shape: {data.shape}")
            ax = axes[i // 2, i % 2]
            ax.imshow(data, aspect='auto', cmap="bwr", vmin=-1000, vmax=1000, origin='lower')
            ax.set_title(f"{titles[i]} TPC{tpc} Plane{plane_suffix} (Event {event_idx})")
            ax.set_xlabel("Channels")
            ax.set_ylabel("Ticks")

    # Plot the predicted image.
    if predicted_image is not None and np.any(predicted_image):
        print(f"Plotting Predicted Image for TPC {tpc} Plane {plane_suffix} Event {event_idx}, shape: {predicted_image.shape}")
        ax = axes[2, 1]
        ax.imshow(predicted_image, aspect='auto', cmap="bwr", vmin=-1000, vmax=1000, origin='lower')
        ax.set_title(f"Predicted Image TPC{tpc} Plane{plane_suffix} (Event {event_idx})")
        ax.set_xlabel("Channels")
        ax.set_ylabel("Ticks")
    else:
        print(f"Predicted image is empty or None for TPC {tpc} Plane {plane_suffix}")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

# Iterate through all subdirectories in the main folder and create a PDF of the plots.
with PdfPages("ICARUS_images_model_nomMC_predict_opaqueMC.pdf") as pdf:
    print("Starting data loading and plotting...")
    
    for folder_name in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder_name)
        print(folder_path)
        
        if os.path.isdir(folder_path):
            for tpc in tpcs:
                for plane, plane_suffix in plane_suffix_map.items():
                    true_file_path = os.path.join(folder_path, f"tpc{tpc}_plane{plane_suffix}_tru.h5")
                    rec_file_path = os.path.join(folder_path, f"tpc{tpc}_plane{plane_suffix}_rec.h5")
                    print(true_file_path)
                    print(rec_file_path)
                    
                    if os.path.exists(true_file_path) and os.path.exists(rec_file_path):
                        print(f"\nProcessing files:\n - True: {true_file_path}\n - Rec: {rec_file_path}")
                        
                        with h5py.File(true_file_path, 'r') as true_file, h5py.File(rec_file_path, 'r') as rec_file:
                            events = list(true_file.keys())
                            if 0 <= event_to_plot < len(events):
                                event = events[event_to_plot]
                                print(f"Loading event: {event}")

                                true_data = true_file[event]['frame_deposplat'][:]
                                rec_data_looseLf = rec_file[event]['frame_looseLf'][:]
                                rec_data_mp3 = rec_file[event]['frame_mp3ROI'][:]
                                rec_data_mp2 = rec_file[event]['frame_mp2ROI'][:]
                                rec_data_gauss = rec_file[event]['frame_gauss'][:]

                                data_dict = {
                                    "DepoSplat": true_data,
                                    "LooseLF": rec_data_looseLf,
                                    "MP3ROI": rec_data_mp3,
                                    "MP2ROI": rec_data_mp2,
                                    "Gauss": rec_data_gauss
                                }

                                # Choose the correct model based on the plane suffix.
                                predicted_image = model_selection(rec_file_path, "U" if plane_suffix == "0" else "V")

                                # Plot all data including the predicted image.
                                plot_data(tpc, plane, plane_suffix, data_dict, event_to_plot, predicted_image, pdf)
                            else:
                                print(f"Event index {event_to_plot} is out of range for TPC {tpc}, Plane {plane}.")
                    else:
                        print(f"Files for TPC {tpc}, Plane {plane} not found in folder {folder_name}.")

print("All images have been saved to a single PDF.")


