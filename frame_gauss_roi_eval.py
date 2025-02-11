import h5py
import numpy as np

# Define file paths
true_mask_file = "/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/samples/opaqueMC/h5s_inspected/tpc0_plane1_tru.h5"
rec_mask_file = "/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/samples/opaqueMC/h5s_inspected/tpc0_plane1_rec.h5"

n_channel=2500
def get_binary_mask(file_path, dataset_name, key, threshold):
    """ Loads an HDF5 dataset and applies thresholding to create a binary mask. """
    with h5py.File(file_path, "r") as f:
        if dataset_name in f and key in f[dataset_name]:
            data = np.array(f[dataset_name][key])
            binary_mask = (data > threshold).astype(np.uint8)  # Convert to 0s and 1s
            print(f"Binary mask for {key} (channel n_channel) after thresholding:\n", binary_mask[:, n_channel])
            return binary_mask
        else:
            return None

def detect_rois(binary_mask):
    """ Detects contiguous ROIs in each channel and returns a list of (start, end) indices. """
    rois = []
    for ich in range(binary_mask.shape[1]):  # Loop over channels
        channel_data = binary_mask[:, ich]
        start, end = None, None
        rois_per_channel = []
        
        for it in range(len(channel_data)):
            if channel_data[it] > 0:
                if start is None:
                    start = it
                end = it
            else:
                if start is not None:
                    rois_per_channel.append((start, end))
                    start, end = None, None
        
        if start is not None:
            rois_per_channel.append((start, end))
        rois.append(rois_per_channel)
    
    if len(rois[n_channel]) > 0:
        print(f"Detected ROIs for channel n_channel: {rois[n_channel]}")
    return rois

def compute_roi_metrics(true_rois, pred_rois):
    """ Computes ROI Efficiency and Purity using DNN evaluation logic. """
    true_detected = 0
    pred_detected = 0
    total_true = sum(len(roi_list) for roi_list in true_rois)
    total_pred = sum(len(roi_list) for roi_list in pred_rois)
    
    for ich in range(len(true_rois)):
        for t_start, t_end in true_rois[ich]:
            if any(p_start <= t_end and p_end >= t_start for p_start, p_end in pred_rois[ich]):
                true_detected += 1
                if ich == n_channel:
                    print(f"Match found for True ROI {t_start}-{t_end} in channel n_channel")
        
        for p_start, p_end in pred_rois[ich]:
            if any(t_start <= p_end and t_end >= p_start for t_start, t_end in true_rois[ich]):
                pred_detected += 1
                if ich == n_channel:
                    print(f"Match found for Predicted ROI {p_start}-{p_end} in channel n_channel")
    
    roi_efficiency = true_detected / total_true if total_true > 0 else 0
    roi_purity = pred_detected / total_pred if total_pred > 0 else 0
    
    print(f"Total True ROIs: {total_true}, Detected True ROIs: {true_detected}")
    print(f"Total Predicted ROIs: {total_pred}, Correct Predicted ROIs: {pred_detected}")
    return roi_efficiency, roi_purity

# Load binary masks
true_mask = get_binary_mask(true_mask_file, "4", "frame_deposplat", threshold=0)
rec_mask = get_binary_mask(rec_mask_file, "4", "frame_gauss", threshold=0.5)

if true_mask is not None and rec_mask is not None:
    # Detect ROIs
    true_rois = detect_rois(true_mask)
    pred_rois = detect_rois(rec_mask)
    
    # Compute ROI efficiency and purity
    roi_efficiency, roi_purity = compute_roi_metrics(true_rois, pred_rois)
    
    print(f"ROI Efficiency: {roi_efficiency:.4f}")
    print(f"ROI Purity: {roi_purity:.4f}")
