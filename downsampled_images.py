import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

from utils import h5_utils as h5u  # same utility module used by HDF5Dataset

#############################
# Adjustable parameters
#############################
TPCS = [0, 1, 2, 3]       # TPC indices
PLANES = [0, 1]           # Plane indices (0: wires=0..2112, 1: wires=0..5600)
REBIN = [1, 8]            # [wire_downsample, time_downsample]
Z_SCALE = 2000
TRUTH_TH = 100
Y_RANGE = [0, 4096]       # Original time dimension; after rebin time becomes 4096/8 = 512

# Main XRootD folder containing input files – update to your actual location
MAIN_XROOTD = (
    "root://fndcadoor.fnal.gov:1094/"
    "pnfs/fnal.gov/usr/sbnd/scratch/users/gputnam/omnidetector-3/nugen/"
    "optfilter-bnb-nomMC-5/59993016_10/h5s_inspected/"
)

# Output files
OUTPUT_PDF = "downsampled_all_events.pdf"
OUTPUT_H5 = "downsampled_all_events.h5"


#############################
# Helper functions
#############################
def get_x_range(plane_id):
    """
    Return the wire dimension range for the given plane.
    Plane 0 -> [0,2112]
    Plane 1 -> [0,5600]
    """
    if plane_id == 0:
        return [0, 2112]
    elif plane_id == 1:
        return [0, 5600]
    else:
        raise ValueError("Invalid plane_id (use 0 or 1)")

def list_frame_datasets(h5_path, group_idx):
    """
    Return all dataset names within group group_idx that start with 'frame_'.
    """
    dsets = []
    with h5py.File(h5_path, "r") as f:
        grp_str = str(group_idx)
        if grp_str in f:
            for name in f[grp_str].keys():
                if name.startswith("frame_"):
                    dsets.append(name)
    return dsets

def list_all_groups(h5_path):
    """
    Return a sorted list of numeric group indices for the file at h5_path.
    """
    groups = []
    if not os.path.exists(h5_path):
        print(f"File {h5_path} not found.")
        return groups
    try:
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                if key.isdigit():
                    groups.append(int(key))
        groups.sort()
    except OSError as e:
        print(f"Could not open file {h5_path}: {e}")
    return groups

def save_event_data(out_h5, tpc, plane_id, group_idx, rec_tags, tru_tags, rec_imgs, tru_imgs):
    """
    Save the downsampled data for one event into the output HDF5 file.
    Folder structure:
       /tpc{tpc}/plane{plane_id}/event{group_idx}/rec/   <-- rec data datasets
       /tpc{tpc}/plane{plane_id}/event{group_idx}/tru/   <-- tru data datasets
    """
    tpc_grp_name = f"tpc{tpc}"
    plane_grp_name = f"plane{plane_id}"
    event_grp_name = f"event{group_idx}"

    # Create groups as needed
    if tpc_grp_name not in out_h5:
        tpc_grp = out_h5.create_group(tpc_grp_name)
    else:
        tpc_grp = out_h5[tpc_grp_name]

    if plane_grp_name not in tpc_grp:
        plane_grp = tpc_grp.create_group(plane_grp_name)
    else:
        plane_grp = tpc_grp[plane_grp_name]

    event_grp = plane_grp.create_group(event_grp_name)
    rec_grp = event_grp.create_group("rec")
    tru_grp = event_grp.create_group("tru")

    # Save rec data: rec_tags and rec_imgs (assumed shape: (N_rec, wires, time))
    for i, tag in enumerate(rec_tags):
        # Save rec_imgs[i] as dataset with the same name (no transpose here)
        rec_grp.create_dataset(tag, data=rec_imgs[i])
    # Save tru data
    for j, tag in enumerate(tru_tags):
        tru_grp.create_dataset(tag, data=tru_imgs[j])

def plot_event_page(rec_file_path, tru_file_path, group_idx, plane_id, tpc, pdf, out_h5):
    """
    For a given event group, load downsampled rec and tru datasets,
    plot them, and save the arrays to out_h5.
    """
    x_range = get_x_range(plane_id)

    # List available datasets in this group
    rec_tags = list_frame_datasets(rec_file_path, group_idx)
    tru_tags = list_frame_datasets(tru_file_path, group_idx)

    # Load rec data (downsampled) using h5u.get_chw_imgs
    if rec_tags:
        rec_list = list(
            h5u.get_chw_imgs(
                rec_file_path,
                [group_idx],
                rec_tags,
                REBIN,
                x_range,
                Y_RANGE,
                Z_SCALE
            )
        )
        rec_imgs = np.array(rec_list[0])
    else:
        rec_imgs = np.array([])

    # Load tru data using h5u.get_masks
    if tru_tags:
        tru_list = list(
            h5u.get_masks(
                tru_file_path,
                [group_idx],
                tru_tags,
                REBIN,
                x_range,
                Y_RANGE,
                TRUTH_TH
            )
        )
        tru_imgs = np.array(tru_list[0])
    else:
        tru_imgs = np.array([])

    # If no data, produce a placeholder plot
    if rec_imgs.size == 0 and tru_imgs.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No frame_* datasets found in event {group_idx}", ha="center", va="center")
        ax.set_title(f"TPC={tpc}, Plane={plane_id}, Event={group_idx}")
        pdf.savefig(fig)
        plt.close(fig)
        return

    # Expand dims if needed
    if rec_imgs.ndim == 2:
        rec_imgs = rec_imgs[None, ...]
    if tru_imgs.ndim == 2:
        tru_imgs = tru_imgs[None, ...]

    N_rec = rec_imgs.shape[0] if rec_imgs.ndim == 3 else 0
    N_tru = tru_imgs.shape[0] if tru_imgs.ndim == 3 else 0
    total_subplots = N_rec + N_tru
    if total_subplots < 1:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data after expand dims?", ha="center", va="center")
        ax.set_title(f"TPC={tpc}, Plane={plane_id}, Event={group_idx}")
        pdf.savefig(fig)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, total_subplots, figsize=(5*total_subplots, 5))
    if total_subplots == 1:
        axes = [axes]
    idx_plot = 0

    # For plotting, we want to show wires on x-axis and time on y-axis.
    # Our arrays are in shape (channels, wires, time); transpose each to (time, wires)
    for i, tag in enumerate(rec_tags):
        channel_data = rec_imgs[i].T  # (time, wires)
        time_size, wire_size = channel_data.shape
        ax = axes[idx_plot]
        idx_plot += 1
        ax.imshow(channel_data, origin="lower", cmap="gray", aspect="auto")
        ax.set_title(f"TPC={tpc}, Pln={plane_id}, Evt={group_idx}\nRec: {tag}")
        ax.set_xlabel(f"Wires (0..{wire_size})")
        ax.set_ylabel(f"Time (0..{time_size})")

    for j, tag in enumerate(tru_tags):
        channel_data = tru_imgs[j].T  # (time, wires)
        time_size, wire_size = channel_data.shape
        ax = axes[idx_plot]
        idx_plot += 1
        ax.imshow(channel_data, origin="lower", cmap="inferno", alpha=0.5, aspect="auto")
        ax.set_title(f"TPC={tpc}, Pln={plane_id}, Evt={group_idx}\nTru: {tag}")
        ax.set_xlabel(f"Wires (0..{wire_size})")
        ax.set_ylabel(f"Time (0..{time_size})")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Save the downsampled arrays to the output h5 file with the desired folder structure.
    save_event_data(out_h5, tpc, plane_id, group_idx, rec_tags, tru_tags, rec_imgs, tru_imgs)


def main():
    """
    Iterates over all TPCs and planes.
    For each TPC-plane, gets all event groups and, for each event,
    downsamples and plots the frame_* datasets, and saves them into an HDF5 file.
    """
    # Open the output HDF5 file for writing the downsampled data.
    with h5py.File(OUTPUT_H5, "w") as out_h5, PdfPages(OUTPUT_PDF) as pdf:
        for tpc in TPCS:
            for plane_id in PLANES:
                rec_file_path = os.path.join(MAIN_XROOTD, f"tpc{tpc}_plane{plane_id}_rec.h5")
                tru_file_path = os.path.join(MAIN_XROOTD, f"tpc{tpc}_plane{plane_id}_tru.h5")
                # Get the union of all numeric group indices from both files.
                rec_groups = list_all_groups(rec_file_path)
                tru_groups = list_all_groups(tru_file_path)
                all_groups = sorted(set(rec_groups).union(set(tru_groups)))
                print(f"\nTPC={tpc} Plane={plane_id} -> Event groups: {all_groups}")
                for group_idx in all_groups:
                    plot_event_page(rec_file_path, tru_file_path, group_idx, plane_id, tpc, pdf, out_h5)
    print(f"\nAll TPC-plane-event pages saved to {OUTPUT_PDF} and downsampled data saved to {OUTPUT_H5}.")


def list_all_groups(h5_path):
    """
    Return a sorted list of numeric groups for the file at h5_path.
    """
    groups = []
    if not os.path.exists(h5_path):
        print(f"File {h5_path} not found.")
        return groups
    try:
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                if key.isdigit():
                    groups.append(int(key))
        groups.sort()
    except OSError as e:
        print(f"Could not open file {h5_path}: {e}")
    return groups


if __name__ == "__main__":
    main()
