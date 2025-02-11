import argparse
import os
import json
import numpy as np
import torch
from tqdm import tqdm

from unet import UNet
from uresnet import UResNet
from nestedunet import NestedUNet
from eval_util_trad import eval_eff_pur
from utils import h5_utils as h5u

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored (default: 'MODEL.pth')")
    parser.add_argument('--config', '-c', default='config-nomMC_U.json',
                        help="Path to the config JSON file (default: 'config-nomMC_U.json')")
    parser.add_argument('--gpu', '-g', action='store_true',
                        help="Use CUDA for evaluation", default=True)
    parser.add_argument('--maskthreshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white", default=0.5)
    parser.add_argument('--output', '-o', metavar='OUTPUT', default='eval_results',
                        help="Output filename for evaluation results")
    parser.add_argument('--range', '-r', type=int, nargs=2,
                        help="File range to be processed (e.g., '--range 0 50')", default=[0, 10])
    return parser.parse_args()

# def read_config(config_path):
#     with open(config_path, 'r') as f:
#         config = json.load(f)
#     return config

import h5py

def read_config(config_path):
    """Reads the JSON config file and loads dataset file paths."""
    with open(config_path, "r") as f:
        config = json.load(f)

    # Ensure `sample_list` and `target_list` are processed
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

    eval_indices = []  # Only include files with at least 5 datasets

    # Check dataset availability in each file
    valid_samples = []
    valid_targets = []
    
    for fileno, file in enumerate(config["sample"]):
        with h5py.File(file, "r") as h5_file:
            available_groups = list(map(int, h5_file.keys()))
            num_datasets = len(available_groups)

            if num_datasets >= 5:
                # Only keep files that have at least 5 datasets
                valid_samples.append(file)
                valid_targets.append(config["target"][fileno])
                eval_indices.append((fileno, 4))  # Use dataset index 4 for evaluation

    if not valid_samples:
        raise ValueError("No valid files found with at least 5 datasets for evaluation!")

    # Update config with valid files
    config["sample"] = valid_samples
    config["target"] = valid_targets
    config["indices"] = {"eval": eval_indices}

    return config




if __name__ == "__main__":
    args = get_args()

    # Load the configuration file
    config = read_config(args.config)

    # # Set up model and device
    # print("Loading model {}".format(args.model))
    # try:
    #     net = torch.jit.load(args.model)
    #     if args.gpu:
    #         net.cuda()
    #     else:
    #         net.cpu()
    #     print("Loaded TorchScript model successfully.")
    # except Exception as e:
    #     print(f"Error loading TorchScript model: {e}")
    #     net = UNet(len(config["im_tags"]), 1)
    #     if args.gpu:
    #         net.cuda()
    #         net.load_state_dict(torch.load(args.model))
    #     else:
    #         net.cpu()
    #         net.load_state_dict(torch.load(args.model, map_location='cpu'))
    #     print("Loaded model using state_dict successfully.")

    # Load evaluation files from the config
    eval_imgs = config["sample"]
    eval_masks = config["target"]
    rebin = config["scale"]
    x_range = config["x_range"]
    y_range = config["y_range"]
    z_scale = config["z_scale"]
    truth_th = config["truth_th"]
    im_tags = config["im_tags"]
    # im_tags=["Gauss"]
    ma_tags = config["ma_tags"]

    # Ensure the range does not exceed the number of available files
    file_range = list(range(args.range[0], min(args.range[1], len(eval_imgs))))
    id_eval = [4]  # Use only index 4 for evaluation from each file

    # Output directory
    dir_out = 'out-eval/800_1600/'
    os.makedirs(dir_out, exist_ok=True)
    outfile_ep = open(f'{dir_out}/{args.output}.csv', 'w')

    # Initialize accumulators for averages
    total_pixel_eff = 0
    total_pixel_pur = 0
    total_instance_eff = 0
    total_instance_pur = 0
    num_files = len(file_range)

    # Perform evaluation with progress bar
    print("Starting evaluation...")
    for idx in tqdm(file_range, total=num_files):
        img_file = eval_imgs[idx]
        mask_file = eval_masks[idx]

        # Prepare data for evaluation
        # data = zip(
        #     h5u.get_chw_imgs(img_file, id_eval, im_tags, rebin, x_range, y_range, z_scale),
        #     h5u.get_masks(mask_file, id_eval, ma_tags, rebin, x_range, y_range, truth_th)
        # )

        with h5py.File(img_file, 'r') as f:
            frame_gauss = f["4/frame_gauss"][:]  # Load traditional method output
        
        with h5py.File(mask_file, 'r') as f:
            frame_deposplat = f["4/frame_deposplat"][:]  # Load ground truth

        print("Shape of frame_gauss:", frame_gauss.shape)
        print("Shape of frame_deposplat:", frame_deposplat.shape)

        
        # dataset = [(frame_gauss[i], frame_deposplat[i]) for i in range(len(frame_gauss))]
        dataset = [(frame_gauss, frame_deposplat)]


        # Evaluate the current file
        # ep = eval_eff_pur(net, data, args.maskthreshold, args.gpu)
        ep = eval_eff_pur("trad", dataset, args.maskthreshold, args.gpu)
        

        # Accumulate metrics
        total_pixel_eff += ep[0]
        total_pixel_pur += ep[1]
        total_instance_eff += ep[2]
        total_instance_pur += ep[3]

        # Write individual results to the output file
        outfile_ep.write(f'{idx}, {ep[0]:.4f}, {ep[1]:.4f}, {ep[2]:.4f}, {ep[3]:.4f}\n')

    # Calculate and display averages
    avg_pixel_eff = total_pixel_eff / num_files
    avg_pixel_pur = total_pixel_pur / num_files
    avg_instance_eff = total_instance_eff / num_files
    avg_instance_pur = total_instance_pur / num_files

    print("\nEvaluation Complete.")
    print(f"Average Pixel Efficiency: {avg_pixel_eff:.4f}")
    print(f"Average Pixel Purity: {avg_pixel_pur:.4f}")
    print(f"Average Instance Efficiency (ROI Efficiency): {avg_instance_eff:.4f}")
    print(f"Average Instance Purity (ROI Purity): {avg_instance_pur:.4f}")

    # Save average metrics to the output file
    outfile_ep.write(f'Average, {avg_pixel_eff:.4f}, {avg_pixel_pur:.4f}, {avg_instance_eff:.4f}, {avg_instance_pur:.4f}\n')
    outfile_ep.close()





