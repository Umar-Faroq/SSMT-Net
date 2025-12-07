import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from mim_transunet.datasets import load_data, NoduleGlandDataset
from mim_transunet.models import get_network


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test MIM-TransUNet model on the TN3K test set"
    )

    parser.add_argument(
        "--data_root_tn3k",
        type=str,
        default="data/TN3K",
        help="Root folder for TN3K dataset",
    )
    parser.add_argument(
        "--data_root_tg3k",
        type=str,
        default="data/tg3k",
        help="Root folder for TG3K dataset",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for testing",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model_recon_dualenc.pth",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="test_results.json",
        help="Path to save test metrics",
    )

    return parser.parse_args()


def compute_dice_iou(preds_binary, masks):
    intersection = (preds_binary * masks).sum(dim=(2, 3))
    union = preds_binary.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))

    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    iou = intersection / (union - intersection + 1e-6)

    return dice.cpu().numpy(), iou.cpu().numpy()


def test_model(
    net,
    test_loader,
    device="cuda",
    checkpoint_path="checkpoints/best_model_recon_dualenc.pth",
    results_path="test_results.json",
):
    print(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    all_dice = []
    all_iou = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images, masks, *_ = batch
            images = images.to(device)
            masks = masks.to(device)

            outputs = net(images)
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                seg_outputs = outputs[1]
            else:
                seg_outputs = outputs

            probs = torch.sigmoid(seg_outputs)
            preds_binary = (probs > 0.5).float()

            dice_batch, iou_batch = compute_dice_iou(preds_binary, masks)
            all_dice.extend(dice_batch)
            all_iou.extend(iou_batch)

    all_dice = np.array(all_dice)
    all_iou = np.array(all_iou)

    dice_mean = float(all_dice.mean())
    dice_std = float(all_dice.std())
    iou_mean = float(all_iou.mean())
    iou_std = float(all_iou.std())

    print(f"\nTest Metrics (Mean ± Std)")
    print(f"Dice: {dice_mean:.6f} ± {dice_std:.6f}")
    print(f"IoU : {iou_mean:.6f} ± {iou_std:.6f}\n")

    results = {
        "Dice_mean": dice_mean,
        "Dice_std": dice_std,
        "IoU_mean": iou_mean,
        "IoU_std": iou_std,
        "Dice_per_image": all_dice.tolist(),
        "IoU_per_image": all_iou.tolist(),
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to: {results_path}")
    return results


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_image_folder = os.path.join(args.data_root_tn3k, "test-image")
    test_mask_folder = os.path.join(args.data_root_tn3k, "test-mask")

    gland_image_folder = os.path.join(args.data_root_tg3k, "thyroid-image")
    gland_mask_folder = os.path.join(args.data_root_tg3k, "thyroid-mask")

    print("TN3K test image folder:", test_image_folder)
    print("TN3K test mask folder :", test_mask_folder)
    print("TG3K gland image folder:", gland_image_folder)

    X_test, Y_test, area_test = load_data(test_image_folder, test_mask_folder, args.img_size)
    X_gland, Y_gland, area_gland = load_data(gland_image_folder, gland_mask_folder, args.img_size)

    test_dataset = NoduleGlandDataset(
        (X_test, Y_test, area_test),
        (X_gland, Y_gland, area_gland),
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    net = get_network(
        vit_name="R50-ViT-B_16",
        img_size=args.img_size,
        num_classes=2,
        n_skip=3,
        vit_patches_size=16,
    )

    os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)
    test_model(
        net,
        test_loader,
        device=device,
        checkpoint_path=args.checkpoint,
        results_path=args.results_path,
    )


if __name__ == "__main__":
    main()
