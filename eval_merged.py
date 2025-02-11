#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
import torch
from tqdm import tqdm
import h5py

# Import the utility for loading images and masks.
from utils import h5_utils as h5u

# Import network definitions in case we need to load via state_dict.
from unet import UNet
from uresnet import UResNet
from nestedunet import NestedUNet


def get_args():
    parser = argparse.ArgumentParser(description="Merged evaluation of traditional and DNN ROI metrics")
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="File in which the model is stored (default: 'MODEL.pth')")
    parser.add_argument('--config', '-c', default='config-nomMC_U.json',
                        help="Path to the config JSON file (default: 'config-nomMC_U.json')")
    parser.add_argument('--gpu', '-g', action='store_true',
                        help="Use CUDA for evaluation", default=True)
    parser.add_argument('--maskthreshold', '-t', type=float, default=0.5,
                        help="Minimum probability value to consider a mask pixel white (default: 0.5)")
    parser.add_argument('--output', '-o', metavar='OUTPUT', default='eval_results',
                        help="Output filename (CSV) for evaluation results (default: 'eval_results')")
    parser.add_argument('--range', '-r', type=int, nargs=2, default=[0, 10],
                        help="File range to be processed (default: [0, 10])")
    parser.add_argument('--debug_channel', '-d', type=int, default=None,
                        help="Wire channel index (after final transposition) for which to print debug info")
    return parser.parse_args()


def read_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    if "sample_list" in config:
        with open(config["sample_list"], "r") as f:
            config["sample"] = [line.strip() for line in f if line.strip()]
    else:
        raise KeyError("'sample_list' is missing in the configuration file")

    if "target_list" in config:
        with open(config["target_list"], "r") as f:
            config["target"] = [line.strip() for line in f if line.strip()]
    else:
        raise KeyError("'target_list' is missing in the configuration file")

    eval_indices = []
    valid_samples = []
    valid_targets = []

    for fileno, file in enumerate(config["sample"]):
        with h5py.File(file, "r") as h5_file:
            available_groups = list(map(int, h5_file.keys()))
            num_datasets = len(available_groups)
            if num_datasets >= 5:
                valid_samples.append(file)
                valid_targets.append(config["target"][fileno])
                eval_indices.append((fileno, 4))

    if not valid_samples:
        raise ValueError("No valid files found with at least 5 datasets for evaluation!")

    config["sample"] = valid_samples
    config["target"] = valid_targets
    config["indices"] = {"eval": eval_indices}
    return config


def eval_pixel(f0, f1, th0=0.0, th1=0.5):
    """
    Evaluate pixel-level efficiency or purity by thresholding f0 and f1.
    shape(f0) = shape(f1) = (wires, time).
    """
    # print(f"[DEBUG] eval_pixel: shape(f0)={f0.shape}, shape(f1)={f1.shape}")
    # print(f"[DEBUG] eval_pixel: thresholds: th0={th0}, th1={th1}")
    # print(f"[DEBUG] eval_pixel BEFORE threshold: "
    #       f"f0_min={f0.min():.2f}, f0_max={f0.max():.2f}, "
    #       f"f1_min={f1.min():.2f}, f1_max={f1.max():.2f}")

    f0m = f0.copy()
    f1m = f1.copy()
    f0m[f0m <= th0] = 0
    f0m[f0m > th0] = 1
    f1m[f1m <= th1] = 0
    f1m[f1m > th1] = 1

    # print(f"[DEBUG] eval_pixel AFTER threshold: "
    #       f"f0m_min={f0m.min()}, f0m_max={f0m.max()}, "
    #       f"f1m_min={f1m.min()}, f1m_max={f1m.max()}")

    num = np.count_nonzero(np.logical_and(f0m, f1m))
    den = np.count_nonzero(f0m)
    if den <= 0:
        return 0
    return num / den


def eval_roi(f0, f1, th0=0.0, th1=0.5, debug_channel=None, debug_tag=""):
    """
    Evaluate ROI-level efficiency or purity by thresholding f0 as denominator, f1 as numerator.
    shape(f0) = shape(f1) = (wires, time).

    We scan along the time axis for each wire to find contiguous runs of 1 in f0,
    and check if f1 has any 1's in those runs.
    """
    # print(f"[DEBUG] eval_roi: shape(f0)={f0.shape}, shape(f1)={f1.shape}")
    # print(f"[DEBUG] eval_roi: thresholds: th0={th0}, th1={th1}")
    # print(f"[DEBUG] eval_roi BEFORE threshold: "
    #       f"f0_min={f0.min():.2f}, f0_max={f0.max():.2f}, "
    #       f"f1_min={f1.min():.2f}, f1_max={f1.max():.2f}")

    f0m = f0.copy()
    f1m = f1.copy()
    f0m[f0m <= th0] = 0
    f0m[f0m >  th0] = 1
    f1m[f1m <= th1] = 0
    f1m[f1m >  th1] = 1

    # print(f"[DEBUG] eval_roi AFTER threshold: "
    #       f"f0m_min={f0m.min()}, f0m_max={f0m.max()}, "
    #       f"f1m_min={f1m.min()}, f1m_max={f1m.max()}")

    num = 0
    den = 0

    # Outer loop: wire
    for wire in range(f0m.shape[0]):  # 0..5599
        start = None
        # Inner loop: time
        for t in range(f0m.shape[1]):  # 0..511
            if f0m[wire, t] == 1:
                if start is None:
                    start = t
            else:
                if start is not None:
                    # We found a contiguous run from [start, t) along time for this wire
                    den += 1
                    # Check overlap in f1
                    if np.any(f1m[wire, start:t] == 1):
                        num += 1
                    start = None

        # If we ended at last time tick still "in" a run
        if start is not None:
            den += 1
            if np.any(f1m[wire, start:f0m.shape[1]] == 1):
                num += 1
            start = None

    if den <= 0:
        return 0
    return num / den


def get_contiguous_intervals_along_time(mask, wire_index):
    """
    For debugging: given a binary mask of shape (wires, time),
    return a list of contiguous intervals (start, end) in time for the specified wire.
    """
    if wire_index >= mask.shape[0]:
        return []
    row = mask[wire_index, :]
    intervals = []
    start = None
    for t in range(len(row)):
        if row[t] == 1:
            if start is None:
                start = t
        else:
            if start is not None:
                intervals.append((start, t))
                start = None
    if start is not None:
        intervals.append((start, len(row)))
    return intervals


def eval_eff_pur(net, dataset, th=0.5, gpu=False, debug_channel=None, debug_tag=""):
    """
    Evaluate both pixel and ROI efficiency and purity over the given dataset.

    For the traditional method, pass net="trad" so that the input image is treated
    directly as the prediction. Otherwise, we run the net on the input image.

    We transpose the arrays so that shape=(wires, time). If they end up as (512,5600),
    we flip them. Then we run eval_pixel and eval_roi with loops that expect
    the shape to be (wires, time).
    """
    eff_pix = 0
    pur_pix = 0
    eff_roi = 0
    pur_roi = 0
    for i, (img, mask_true) in enumerate(dataset):
        # 1) Get the prediction
        if net == "trad":
            mask_pred = img
        else:
            img_tensor = torch.from_numpy(img).unsqueeze(0)
            if gpu:
                img_tensor = img_tensor.cuda()
            with torch.no_grad():
                mask_pred = net(img_tensor).squeeze().cpu().numpy()

        # 2) We want shape(wires, time).
        # Transpose so that axis 0 is wires, axis 1 is time.
        # If it ends up (time, wires) = (512,5600), we flip it.
        # But we expect (5600,512) => wires=5600, time=512
        mask_true = np.transpose(mask_true, [1, 0])
        mask_pred = np.transpose(mask_pred, [1, 0])
        # If they come out (512, 5600) => (time, wires),
        # then shape[0]<shape[1], flip them
        if mask_true.shape[0] < mask_true.shape[1]:
            mask_true = np.transpose(mask_true, [1, 0])
        if mask_pred.shape[0] < mask_pred.shape[1]:
            mask_pred = np.transpose(mask_pred, [1, 0])

        # 3) Print debug info about shape
        if debug_channel is not None:
            print(f"DEBUG ({debug_tag}): Final shape(wires,time) = {mask_true.shape}")

            # If the user wants intervals for a channel=wire_index
            if debug_channel < mask_true.shape[0]:
                # We'll do a "temp" threshold with 0.5 for ground truth, th for predictions
                # just to get an idea of intervals
                # Or use the same logic we do in eval_roi
                gt_thresh = mask_true.copy()
                pred_thresh = mask_pred.copy()
                gt_thresh[gt_thresh <= 0.5] = 0
                gt_thresh[gt_thresh > 0.5] = 1
                pred_thresh[pred_thresh <= th] = 0
                pred_thresh[pred_thresh > th] = 1

                gt_intervals = get_contiguous_intervals_along_time(gt_thresh, debug_channel)
                pred_intervals = get_contiguous_intervals_along_time(pred_thresh, debug_channel)
                print(f"---------- ROI Debug: {debug_tag} Branch, Wire={debug_channel} ----------")
                print("  Ground Truth contiguous intervals (time):", gt_intervals)
                print("  Predicted contiguous intervals (time):", pred_intervals)

        # 4) Evaluate
        # We'll keep the same threshold strategy: ground truth=0.5, prediction=th
        # This is the logic that you might want to adjust if you prefer a different threshold for "trad" vs DNN
        # e.g. eff_pix += eval_pixel(mask_true, mask_pred, 0.5, your_threshold_for_trad)
        eff_pix += eval_pixel(mask_true, mask_pred, 0.5, th)
        pur_pix += eval_pixel(mask_pred, mask_true, th, 0.5)
        eff_roi += eval_roi(mask_true, mask_pred, 0.5, th, debug_channel, debug_tag)
        pur_roi += eval_roi(mask_pred, mask_true, th, 0.5, debug_channel, debug_tag)

    n = i + 1
    return [eff_pix / n, pur_pix / n, eff_roi / n, pur_roi / n]


# ---------------- Main Evaluation Routine ----------------

if __name__ == "__main__":
    args = get_args()
    config = read_config(args.config)

    # Load evaluation parameters from the config file
    eval_imgs  = config["sample"]
    eval_masks = config["target"]
    rebin      = config["scale"]     # [1,8]
    x_range    = config["x_range"]   # e.g. [0,5600]
    y_range    = config["y_range"]   # e.g. [0,4096]
    z_scale    = config["z_scale"]
    truth_th   = config["truth_th"]
    im_tags    = config["im_tags"]
    ma_tags    = config["ma_tags"]

    # Load the model
    print(f"Loading model {args.model}...")
    try:
        net = torch.jit.load(args.model)
        if args.gpu:
            net.cuda()
        else:
            net.cpu()
        print("Loaded TorchScript model successfully.")
    except Exception as e:
        print(f"Error loading TorchScript model: {e}")
        net = UNet(len(im_tags), 1)
        if args.gpu:
            net.cuda()
            net.load_state_dict(torch.load(args.model))
        else:
            net.cpu()
            net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Loaded model using state_dict successfully.")

    file_range = list(range(args.range[0], min(args.range[1], len(eval_imgs))))
    id_eval = [4]

    # Output directory
    dir_out = 'out-eval/merged/hpsee-nomMC'
    os.makedirs(dir_out, exist_ok=True)
    outfile_path = os.path.join(dir_out, f"{args.output}.csv")
    outfile = open(outfile_path, 'w')
    outfile.write("File, Trad_Pixel_Eff, Trad_Pixel_Pur, Trad_ROI_Eff, Trad_ROI_Pur, "
                  "DNN_Pixel_Eff, DNN_Pixel_Pur, DNN_ROI_Eff, DNN_ROI_Pur\n")

    total_trad = np.zeros(4)
    total_dnn  = np.zeros(4)
    num_files = len(file_range)

    print("Starting evaluation...")
    for idx in tqdm(file_range, total=len(file_range)):
        img_file = eval_imgs[idx]
        mask_file = eval_masks[idx]

        # ----- Traditional -----
        with h5py.File(img_file, 'r') as f:
            frame_gauss_full = f["4/frame_gauss"][:]
        with h5py.File(mask_file, 'r') as f:
            frame_deposplat_full = f["4/frame_deposplat"][:]

        # print(f"Native resolution of frame_gauss_full: {frame_gauss_full.shape}")
        # print(f"Native resolution of frame_deposplat_full: {frame_deposplat_full.shape}")

        # e.g. Crop time dimension to first 4096, keep wires=5600
        cropped_gauss     = frame_gauss_full[0:4096, :]
        cropped_deposplat = frame_deposplat_full[0:4096, :]

        # Rebin time dimension from 4096 => 512
        target_shape_img  = (cropped_gauss.shape[0] // 8, cropped_gauss.shape[1])  # => (512, 5600)
        frame_gauss_ds    = h5u.rebin(cropped_gauss, target_shape_img)
        target_shape_mask = (cropped_deposplat.shape[0] // 8, cropped_deposplat.shape[1])
        frame_deposplat_ds= h5u.rebin(cropped_deposplat, target_shape_mask)

        # Threshold the ground truth if truth_th is set
        if truth_th is not None:
            # truth_th=0
            frame_deposplat_ds[frame_deposplat_ds <= truth_th] = 0
            frame_deposplat_ds[frame_deposplat_ds >  truth_th] = 1

        # Now binarize frame_gauss_ds exactly like ground truth, using truth_th
        if truth_th is not None:
            truth_th_gauss=0
            frame_gauss_ds[frame_gauss_ds == truth_th_gauss] = 0
            frame_gauss_ds[frame_gauss_ds !=  truth_th_gauss] = 1
        

        unique_vals = np.unique(frame_deposplat_ds)
        # print(f"[DEBUG] Ground truth after truth_th binarization: unique values = {unique_vals}")
        # print(f"[DEBUG] Ground truth stats after binarization: min={frame_deposplat_ds.min()}, "
              # f"max={frame_deposplat_ds.max()}")

        unique_vals = np.unique(frame_gauss_ds)
        # print(f"[DEBUG] Traditional after truth_th binarization: unique values = {unique_vals}")
        # print(f"[DEBUG] Traditional stats after binarization: min={frame_gauss_ds.min()}, "
              # f"max={frame_gauss_ds.max()}")

        # Evaluate Traditional
        dataset_trad = [(frame_gauss_ds, frame_deposplat_ds)]
        ep_trad = eval_eff_pur(
            "trad", dataset_trad, args.maskthreshold, args.gpu,
            debug_channel=args.debug_channel, debug_tag="Traditional"
        )

        # ----- DNN -----
        imgs_dnn  = h5u.get_chw_imgs(img_file, id_eval, im_tags, rebin, x_range, y_range, z_scale)
        masks_dnn = h5u.get_masks(mask_file, id_eval, ma_tags, rebin, x_range, y_range, truth_th)
        data_dnn  = list(zip(imgs_dnn, masks_dnn))
        ep_dnn = eval_eff_pur(
            net, data_dnn, args.maskthreshold, args.gpu,
            debug_channel=args.debug_channel, debug_tag="DNN"
        )

        total_trad += np.array(ep_trad)
        total_dnn  += np.array(ep_dnn)

        # Write CSV
        outfile.write(f"{idx}, {ep_trad[0]:.4f}, {ep_trad[1]:.4f}, {ep_trad[2]:.4f}, {ep_trad[3]:.4f}, "
                      f"{ep_dnn[0]:.4f}, {ep_dnn[1]:.4f}, {ep_dnn[2]:.4f}, {ep_dnn[3]:.4f}\n")

    # Final Stats
    avg_trad = total_trad / num_files
    avg_dnn  = total_dnn  / num_files

    print("\nEvaluation Complete.")
    print("Traditional Evaluation:")
    print(f"  Average Pixel Efficiency: {avg_trad[0]:.4f}")
    print(f"  Average Pixel Purity:     {avg_trad[1]:.4f}")
    print(f"  Average ROI Efficiency:   {avg_trad[2]:.4f}")
    print(f"  Average ROI Purity:       {avg_trad[3]:.4f}")

    print("\nDNN Evaluation:")
    print(f"  Average Pixel Efficiency: {avg_dnn[0]:.4f}")
    print(f"  Average Pixel Purity:     {avg_dnn[1]:.4f}")
    print(f"  Average ROI Efficiency:   {avg_dnn[2]:.4f}")
    print(f"  Average ROI Purity:       {avg_dnn[3]:.4f}")

    outfile.write(f"Average, {avg_trad[0]:.4f}, {avg_trad[1]:.4f}, {avg_trad[2]:.4f}, {avg_trad[3]:.4f}, "
                  f"{avg_dnn[0]:.4f}, {avg_dnn[1]:.4f}, {avg_dnn[2]:.4f}, {avg_dnn[3]:.4f}\n")
    outfile.close()


