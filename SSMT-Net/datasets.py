import os
import random
import numpy as np
from glob import glob
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def load_image(path, size):
    """Loads an image, resizes it, converts to grayscale, and normalizes."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    return image


def load_data(image_folder, mask_folder, size):
    """Loads images and corresponding masks, and computes normalized areas."""
    images, masks, areas = [], [], []
    
    image_paths = sorted(glob(os.path.join(image_folder, "*")))
    mask_paths = sorted(glob(os.path.join(mask_folder, "*")))

    image_dict = {os.path.basename(p): p for p in image_paths}
    mask_dict = {os.path.basename(p): p for p in mask_paths}

    for filename in image_dict.keys():
        if filename in mask_dict:
            img = load_image(image_dict[filename], size)
            mask = load_image(mask_dict[filename], size)
            area = np.sum(mask)
            
            images.append(img)
            masks.append(mask)
            areas.append(area)

    areas = np.array(areas)
    if areas.size > 0:
        min_area, max_area = areas.min(), areas.max()
        normalized_areas = (areas - min_area) / (max_area - min_area + 1e-8)
    else:
        print("areas array is empty. Cannot compute min/max.")
        normalized_areas = np.array([])

    return np.array(images), np.array(masks), normalized_areas


class NoduleGlandDataset(Dataset):
    """Dataset for Nodule and Gland images and masks with augmentations."""
    
    def __init__(self, nodule_data, gland_data, augment=False):
        self.nodule_data = nodule_data
        self.gland_data = gland_data
        self.augment = augment

        self.base_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.15),
            transforms.RandomVerticalFlip(p=0.15),
            transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.NEAREST),
        ])

        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.mask_transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.nodule_data[0])

    def __getitem__(self, idx):
        """Loads the image and mask using their file paths."""
        nodule_img, nodule_mask, nodule_area = self.nodule_data[0][idx], self.nodule_data[1][idx], self.nodule_data[2][idx]
        gland_img, gland_mask, gland_area = self.gland_data[0][idx], self.gland_data[1][idx], self.gland_data[2][idx]
        
        nodule_img = Image.fromarray(nodule_img)
        nodule_mask = Image.fromarray(nodule_mask)
        gland_img = Image.fromarray(gland_img)
        gland_mask = Image.fromarray(gland_mask)
        
        if self.augment:
            seed = random.randint(0, 9999)
            torch.manual_seed(seed)
            nodule_img = self.base_transform(nodule_img)
            gland_img = self.base_transform(gland_img)

            torch.manual_seed(seed)
            nodule_mask = self.base_transform(nodule_mask)
            gland_mask = self.base_transform(gland_mask)
        
        nodule_img = self.tensor_transform(nodule_img)
        nodule_mask = self.mask_transform(nodule_mask)
        gland_img = self.tensor_transform(gland_img)
        gland_mask = self.mask_transform(gland_mask)
        
        return nodule_img, nodule_mask, torch.tensor(nodule_area, dtype=torch.float32), gland_img, gland_mask, torch.tensor(gland_area, dtype=torch.float32)
