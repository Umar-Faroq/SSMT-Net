import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from SSMT-Net.datasets import load_data, NoduleGlandDataset
from SSMT-Net.models import get_network, DiceLoss
from SSMT-Net.train_utils import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SSMT-Net for thyroid nodule and gland segmentation"
    )

    # Root folders for datasets (relative to repo or absolute)
    parser.add_argument(
        "--data_root_tn3k",
        type=str,
        default="data/TN3K",
        help="Root folder for TN3K dataset (containing trainval-image, trainval-mask, test-image, test-mask)",
    )
    parser.add_argument(
        "--data_root_tg3k",
        type=str,
        default="data/tg3k",
        help="Root folder for TG3K gland dataset (containing thyroid-image, thyroid-mask)",
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Input image size (square)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoints/best_model_recon_dualenc.pth",
        help="Path to save the best model",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="results.json",
        help="Path to save training/validation metrics JSON",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Define paths for datasets (now relative / configurable)
    # ------------------------------------------------------------------
    trainval_image_folder = os.path.join(args.data_root_tn3k, "trainval-image")
    trainval_mask_folder = os.path.join(args.data_root_tn3k, "trainval-mask")
    test_image_folder = os.path.join(args.data_root_tn3k, "test-image")
    test_mask_folder = os.path.join(args.data_root_tn3k, "test-mask")

    train_gland_img_folder = os.path.join(args.data_root_tg3k, "thyroid-image")
    train_gland_mask_folder = os.path.join(args.data_root_tg3k, "thyroid-mask")

    img_size = args.img_size

    # Sanity print
    print("TN3K train/val image folder:", trainval_image_folder)
    print("TN3K test image folder:", test_image_folder)
    print("TG3K gland image folder:", train_gland_img_folder)

    # ------------------------------------------------------------------
    # 2. Load datasets (same logic as your notebook)
    # ------------------------------------------------------------------
    X_trainval, Y_trainval, area_trainval = load_data(
        trainval_image_folder, trainval_mask_folder, img_size
        )
    X_test, Y_test, area_test = load_data(
        test_image_folder, test_mask_folder, img_size
        )
    X_gland, Y_gland, area_gland = load_data(
        train_gland_img_folder, train_gland_mask_folder, img_size
        )

    # ------------------------------------------------------------------
    # 3. Split trainval into training (80%) and validation (20%)
    # ------------------------------------------------------------------
    X_train, X_valid, Y_train, Y_valid, area_train, area_valid = train_test_split(
        X_trainval,
        Y_trainval,
        area_trainval,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    # ------------------------------------------------------------------
    # 4. Create datasets and dataloaders
    # ------------------------------------------------------------------
    train_dataset = NoduleGlandDataset(
        (X_train, Y_train, area_train),
        (X_gland, Y_gland, area_gland),
        augment=True,
    )
    valid_dataset = NoduleGlandDataset(
        (X_valid, Y_valid, area_valid),
        (X_gland, Y_gland, area_gland),
        augment=False,
    )
    test_dataset = NoduleGlandDataset(
        (X_test, Y_test, area_test),
        (X_gland, Y_gland, area_gland),
        augment=False,
    )

    BATCH_SIZE = args.batch_size

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # ------------------------------------------------------------------
    # 5. Build model, losses, optimizer
    # ------------------------------------------------------------------
    net = get_network(
        vit_name="R50-ViT-B_16",
        img_size=img_size,
        num_classes=2,
        n_skip=3,
        vit_patches_size=16,
    )

    seg_criterion = DiceLoss()
    rec_criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)

    # ------------------------------------------------------------------
    # 6. Train model (internally handles gland-then-nodule phases)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    train_model(
        net,
        train_loader,
        valid_loader,
        seg_criterion,
        rec_criterion,
        optimizer,
        num_epochs=args.num_epochs,
        device=device,
        save_path=args.save_path,
        results_path=args.results_path,
    )

if __name__ == "__main__":
    main()
