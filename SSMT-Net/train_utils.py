import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from .models import DiceLoss  # if you need it here, or keep DiceLoss in models only

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(net, train_loader, valid_loader, seg_criterion, rec_criterion, optimizer, num_epochs=50, device='cuda', save_path="/home/dilab/ext_drive/Thyroid_Nodule_segmentation/WACV_Rebuttal/Test_dir/best_model_recon_dualenc.pth", results_path="results.json"):
    net.to(device)
    # net.load_state_dict(torch.load("pretrained.pth"))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_loss = float("inf")  # Track best validation loss
    best_dice_score = 0.0  # Track best dice score
    area_criterion = nn.MSELoss()  # Define Mean Squared Error loss for area prediction
    # 
    run_name = f"thyroid_exp01_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    for epoch in range(num_epochs):
        # Training code remains unchanged
        net.train()
        epoch_loss = 0.0

        # for images, masks, areas, gland_images, gland_masks, gland_areas in train_loader:
        for batch_idx, (images, masks, areas, gland_images, gland_masks, gland_areas) in enumerate(train_loader):

            images, masks, areas = images.to(device), masks.to(device), areas.to(device)
            gland_images, gland_masks, gland_areas = gland_images.to(device), gland_masks.to(device), gland_areas.to(device)
            optimizer.zero_grad()
            if epoch < 50:
                # --- Phase 1: Train on gland data ---
                # gland_images, gland_masks, gland_areas = gland_images.to(device), gland_masks.to(device), gland_areas.to(device)

                seg_out1, seg_out2, rec_out, area_out = net(gland_images)
                seg_loss = seg_criterion(seg_out1, gland_masks)
                rec_loss = rec_criterion(rec_out, gland_images)
                area_loss = area_criterion(area_out.squeeze(-1), gland_areas)

            else:
                # --- Phase 2: Train on nodule data ---
                images, masks, areas = images.to(device), masks.to(device), areas.to(device)

                seg_out1, seg_out2, rec_out, area_out = net(images)
                seg_loss = seg_criterion(seg_out2, masks)
                rec_loss = rec_criterion(rec_out, images)
                area_loss = area_criterion(area_out.squeeze(-1), areas)

            loss = 0.8 * seg_loss  + 0.1 * rec_loss  + 0.1 * area_loss
            loss.backward()
            optimizer.step()
            # Inside training loop, after optimizer.step()
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + batch_idx)
            
            epoch_loss += loss.item()

        scheduler.step()
        # added this line of code for tensor board
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # ✅ TensorBoard logging for training loss and learning rate
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)



        # Validation Step
        net.eval()
        val_loss = 0.0
        dice_scores = []
        with torch.no_grad():
            for images, masks, areas, gland_images, gland_masks, gland_areas in valid_loader:
                images, masks, areas = images.to(device), masks.to(device), areas.to(device)
                seg_outputs1, seg_outputs2, rec_outputs, area_output = net(images)
                
                seg_loss2 = seg_criterion(seg_outputs2, masks)
                # loss = seg_loss2  # Phase 3 validation
                rec_loss = rec_criterion(rec_outputs, images)
                area_loss = area_criterion(area_output.squeeze(-1), areas)
                loss = 0.8 * seg_loss2 + 0.1 * rec_loss + 0.1 * area_loss
                val_loss += loss.item()
    
                preds2 = torch.sigmoid(seg_outputs2)
                preds2 = (preds2 > 0.5).float()
                intersection2 = (preds2 * masks).sum(dim=(2, 3))
                union2 = preds2.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                dice_score2 = (2. * intersection2 + 1e-6) / (union2 + 1e-6)
                dice_scores.append(dice_score2.mean().item())

                if epoch % 10 == 0:
                    writer.add_images("Inputs", images[:4], epoch)
                    writer.add_images("GroundTruth", masks[:4], epoch)
                    writer.add_images("Predictions", preds2[:4], epoch)

    
        avg_val_loss = val_loss / len(valid_loader)
        avg_dice_score = np.mean(dice_scores)
        # add these lines of code for tensorboard
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Dice/val", avg_dice_score, epoch)
        # writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        print(f"Validation Loss: {avg_val_loss:.6f}, Dice Score: {avg_dice_score:.6f}")
    
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_dice_score = avg_dice_score  # Store the best dice score
            torch.save(net.state_dict(), save_path)
            print(f"✅ Best model saved with Validation Loss: {best_val_loss:.6f}, Dice Score: {avg_dice_score:.6f} ✅")

    print("🏆 Training Complete!")