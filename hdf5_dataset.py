import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import h5_utils as h5u  # Import utility functions

class HDF5Dataset(Dataset):
    def __init__(self, files_img, files_mask, img_tags, mask_tags, indices, rebin, x_range, y_range, z_scale, truth_th):
        self.files_img = files_img  # list of input files
        self.files_mask = files_mask  # list of target files
        assert len(files_img) == len(files_mask)
        self.nfiles = len(files_img)
        self.img_tags = img_tags
        self.mask_tags = mask_tags
        self.indices = indices
        self.rebin = rebin
        self.x_range = x_range
        self.y_range = y_range
        self.z_scale = z_scale
        self.truth_th = truth_th

    def __len__(self):
        return self.nfiles * len(self.indices)

    def __getitem__(self, index):
        # Get file number and event ID within the file
        fileno = index // len(self.indices)
        this_ID = index % len(self.indices)

        # Check if the event ID is within the valid range
        with h5py.File(self.files_img[fileno], 'r') as h5_file:
            available_keys = list(h5_file.keys())
            if this_ID >= len(available_keys):
                raise IndexError(f"Index {this_ID} out of range. Available keys: {available_keys}")

        # Log file information
        # print(f"fileno: {fileno}, this_ID: {this_ID}, file_img: {self.files_img[fileno]}, file_mask: {self.files_mask[fileno]}")

        # Tag replacement logic: No need to replace the "mp3ROI" tag as it's constant
        this_img_tags = self.img_tags
        this_mask_tags = self.mask_tags

        # print(f"Using img_tags: {this_img_tags} and mask_tags: {this_mask_tags}")

        # Load images and masks for the current event
        imgs = list(h5u.get_chw_imgs(self.files_img[fileno], [this_ID], this_img_tags, self.rebin, self.x_range, self.y_range, self.z_scale))[0]
        masks = list(h5u.get_masks(self.files_mask[fileno], [this_ID], this_mask_tags, self.rebin, self.x_range, self.y_range, self.truth_th))[0]

        return torch.tensor(imgs, dtype=torch.float32), torch.tensor(masks, dtype=torch.float32)
