import datetime
import sys
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from unet import UNet
from hdf5_dataset import HDF5Dataset
from eval_util import eval_dice_loss
from dice_loss import dice_coeff
import h5py


def eval_img(net, dataset, gpu=False):
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]

        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(true_mask, torch.Tensor):
            true_mask = true_mask.cpu().numpy()

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]

        return true_mask.detach().cpu().numpy(), mask_pred.detach().cpu().numpy()


def train_net(
    net,
    gpu=False,
    save_cp=False,
    dir_checkpoint="",
    batch_size=2,
    lr=0.1,
    sample="",
    target="",
    sepoch=0,
    nepoch=1,
    img_scale=[1, 10],
    x_range=[0, 1984],
    y_range=[0, 3500],
    z_scale=2000,
    dtype="float32",
    truth_th=100,
    im_tags=["frame_loose_lf1", "frame_mp2_roi1", "frame_mp3_roi1"],
    ma_tags=["frame_deposplat1"],
    indices=None,
):
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)
    outfile_log = open(dir_checkpoint + "/log", "a+")
    outfile_loss = open(dir_checkpoint + "/train-loss.csv", "a+")
    outfile_dice = open(dir_checkpoint + "/train-dice.csv", "a+")
    outfile_eval_loss = open(dir_checkpoint + "/eval-loss.csv", "a+")
    outfile_eval_dice = open(dir_checkpoint + "/eval-dice.csv", "a+")

    DT_STR = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(dir_checkpoint + "/tensorboard/" + DT_STR)

    data_loader_args = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "drop_last": False,
    }

    train_dataset = HDF5Dataset(
        files_img=sample,
        files_mask=target,
        img_tags=im_tags,
        mask_tags=ma_tags,
        indices=indices["train"],
        rebin=img_scale,
        x_range=x_range,
        y_range=y_range,
        z_scale=z_scale,
        truth_th=truth_th,
    )
    train_loader = DataLoader(train_dataset, **data_loader_args)

    val_dataset = HDF5Dataset(
        files_img=sample,
        files_mask=target,
        img_tags=im_tags,
        mask_tags=ma_tags,
        indices=indices["val"],
        rebin=img_scale,
        x_range=x_range,
        y_range=y_range,
        z_scale=z_scale,
        truth_th=truth_th,
    )
    val_loader = DataLoader(val_dataset, **data_loader_args)

    test_dataset = HDF5Dataset(
        files_img=sample,
        files_mask=target,
        img_tags=im_tags,
        mask_tags=ma_tags,
        indices=indices["test"],
        rebin=img_scale,
        x_range=x_range,
        y_range=y_range,
        z_scale=z_scale,
        truth_th=truth_th,
    )
    test_loader = DataLoader(test_dataset, **data_loader_args)

    if sepoch > 0 :
        net = torch.jit.load('{}/CP{}.pth'.format(dir_checkpoint, sepoch-1))

    optimizer = optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
    )
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_dice = 0
    best_val_loss = float("inf")

    for epoch in range(sepoch, sepoch + nepoch):
        print(f"Starting epoch {epoch}/{nepoch}.")

        # Training Phase
        net.train()
        epoch_loss = 0
        for imgs, true_masks in tqdm(train_loader, desc="Training"):
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch} finished! Loss: {epoch_loss:.6f}")
        writer.add_scalar("loss/train", epoch_loss, epoch)
        print(f"{epoch:.4f}, {epoch_loss:.6f}", file=outfile_loss, flush=True)

        torch.jit.save(torch.jit.script(net), dir_checkpoint + 'CP{}.pth'.format(epoch))
        print('Checkpoint e{} saved !'.format(epoch))

        # Validation Phase
        net.eval()
        with torch.no_grad():
            val_dice, val_loss = eval_dice_loss(net, val_loader, criterion, gpu)
            print(
                f"Validation Dice Coeff: {val_dice:.4f}, Loss: {val_loss:.6f}"
            )
            print(
                f"{epoch:.4f}, {val_dice:.6f}", file=outfile_eval_dice, flush=True
            )
            print(
                f"{epoch:.4f}, {val_loss:.6f}", file=outfile_eval_loss, flush=True
            )

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.jit.save(torch.jit.script(net), dir_checkpoint + '/best_dice.pth')
                print("Saved best Dice coefficient model.")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.jit.save(torch.jit.script(net), dir_checkpoint + '/best_loss.pth')
                print("Saved best loss model.")

        # Testing Phase
        with torch.no_grad():
            test_loss = 0
            test_dice = 0
            n_test_batches = 0
            for imgs, true_masks in tqdm(test_loader, desc="Testing"):
                if gpu:
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()

                masks_pred = net(imgs)
                masks_probs_flat = masks_pred.view(-1)
                true_masks_flat = true_masks.view(-1)

                loss = criterion(masks_probs_flat, true_masks_flat)
                test_loss += loss.item()

                dice_score = dice_coeff(masks_pred, true_masks).item()
                test_dice += dice_score

                n_test_batches += 1

            test_loss /= n_test_batches
            test_dice /= n_test_batches
            print(
                f"Test Loss: {test_loss:.6f}, Test Dice Coeff: {test_dice:.4f}"
            )
            writer.add_scalar("loss/test", test_loss, epoch)
            writer.add_scalar("dice/test", test_dice, epoch)

    writer.close()
    print("Training complete")


def read_config(cfgname):
    with open(cfgname, "r") as fin:
        config = json.load(fin)

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

    # Generate train/val/test indices dynamically
    train_indices = []
    val_indices = []
    test_indices = []

    for fileno, file in enumerate(config["sample"]):
        with h5py.File(file, "r") as h5_file:
            available_groups = list(map(int, h5_file.keys()))
            if len(available_groups) == 1:
                train_indices.append((fileno, 0))
            elif len(available_groups) == 2:
                train_indices.append((fileno, 0))
                val_indices.append((fileno, 1))
            elif len(available_groups) == 3:
                train_indices.extend([(fileno, 0), (fileno, 1)])
                val_indices.append((fileno, 2))
            elif len(available_groups) >= 4:
                train_indices.extend([(fileno, 0), (fileno, 1)])
                val_indices.append((fileno, 2))
                test_indices.append((fileno, 3))

    config["indices"] = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    return config



def get_args():
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-c", "--config", help="JSON with script configuration")
    parser.add_option(
        "-g", "--gpu", action="store_true", dest="gpu", default=False
    )
    parser.add_option("-s", "--savecp", dest="savecp", default=False)
    parser.add_option("-l", "--load", dest="load", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args, _ = get_args()
    config = read_config(args.config)
    net = UNet(len(config["im_tags"]), len(config["ma_tags"]))

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print(f"Model loaded from {args.load}")

    if args.gpu:
        net.cuda()

    train_net(
        net=net,
        gpu=args.gpu,
        save_cp=args.savecp,
        dir_checkpoint=config["dir_checkpoint"],
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        sample=config["sample"],
        target=config["target"],
        sepoch=config["start_epoch"],
        nepoch=config["nepoch"],
        img_scale=config["scale"],
        x_range=config["x_range"],
        y_range=config["y_range"],
        z_scale=config["z_scale"],
        dtype=config["dtype"],
        truth_th=config["truth_th"],
        im_tags=config["im_tags"],
        ma_tags=config["ma_tags"],
        indices=config["indices"],
    )
