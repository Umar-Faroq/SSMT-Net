import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from mim_transunet.datasets import load_data, NoduleGlandDataset
from mim_transunet.models import get_network, DiceLoss
from mim_transunet.train_utils import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # ==== Paths (adapt these to your actual folders) ====
    nodule_img_dir = "path/to/nodule/images"
    nodule_mask_dir = "path/to/nodule/masks"
    gland_img_dir  = "path/to/gland/images"
    gland_mask_dir = "path/to/gland/masks"
    img_size = 224

    # ==== Load data ====
    X_nodule, Y_nodule, area_nodule = load_data(nodule_img_dir, nodule_mask_dir, img_size)
    X_gland,  Y_gland,  area_gland  = load_data(gland_img_dir, gland_mask_dir, img_size)

    # You already had train/val/test split in the notebook; copy it here:
    X_trainval, X_test, Y_trainval, Y_test, area_trainval, area_test = train_test_split(
        X_nodule, Y_nodule, area_nodule, test_size=0.2, shuffle=True, random_state=42
    )

    X_train, X_valid, Y_train, Y_valid, area_train, area_valid = train_test_split(
        X_trainval, Y_trainval, area_trainval, test_size=0.2, shuffle=True, random_state=42
    )

    # ==== Datasets & Dataloaders ====
    train_dataset = NoduleGlandDataset((X_train, Y_train, area_train), (X_gland, Y_gland, area_gland), augment=True)
    valid_dataset = NoduleGlandDataset((X_valid, Y_valid, area_valid), (X_gland, Y_gland, area_gland), augment=False)
    test_dataset  = NoduleGlandDataset((X_test,  Y_test,  area_test),  (X_gland, Y_gland, area_gland), augment=False)

    BATCH_SIZE = 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ==== Model ====
    net = get_network(
        vit_name='R50-ViT-B_16',
        img_size=img_size,
        num_classes=2,
        n_skip=3,
        vit_patches_size=16
    )

    # ==== Losses & optimizer ====
    seg_criterion = DiceLoss()
    rec_criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)

    # ==== Train ====
    train_model(
        net,
        train_loader,
        valid_loader,
        seg_criterion,
        rec_criterion,
        optimizer,
        num_epochs=100,
        device=device,
        save_path="checkpoints/best_model_recon_dualenc.pth",
        results_path="results.json"
    )

if __name__ == "__main__":
    main()