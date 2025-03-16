import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from unet_model import UNet #from unet_model.py
from medpy.metric.binary import dc, hd  

# define dataset paths
IMAGE_DIR = "/Users/jojo/Task01_BrainTumour/imagesTr/"
MASK_DIR = "/Users/jojo/Task01_BrainTumour/labelsTr/"

# define dataset class
class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # load MRI scan & corresponding mask
        img_nifti = nib.load(img_path).get_fdata()
        mask_nifti = nib.load(mask_path).get_fdata()

        # choose which modality to use and which slice
        slice_idx = img_nifti.shape[2] // 2  # middle slice
        img = img_nifti[:, :, slice_idx, 0]  # use FLAIR
        mask = mask_nifti[:, :, slice_idx]  # corresponding segmentation mask

        # normalize image pixel value
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        # convert to tensors
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # add a channel dimension
        mask = torch.tensor(mask, dtype=torch.long)  # segmentation labels

        return img, mask

# initialize dataset
dataset = BrainTumorDataset(IMAGE_DIR, MASK_DIR)

# k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

# training parameters
num_epochs = 5

# 5-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n--- Fold {fold+1} ---")
    
    # create training and validation subsets
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # create data loaders
    train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=2, shuffle=False)

    model = UNet(in_channels=1, out_channels=4).to(device) #initialize UNet model
    optimizer = optim.Adam(model.parameters(), lr=1e-4) #initialize optimizer
    loss_fn = nn.CrossEntropyLoss() #choose and initialize loss

    # training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, masks in tqdm(train_loader, desc=f"Training Fold {fold+1}, Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            # forward pass
            preds = model(images)
            loss = loss_fn(preds, masks)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # eval. on validation set
    model.eval()
    dice_scores = []
    hausdorff_dists = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Evaluating Fold {fold+1}"):
            images, masks = images.to(device), masks.to(device)

            # get predictions
            preds = model(images)
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()

            # compute Dice Score & Hausdorff Distance for each sample given
            for i in range(len(images)):
                # convert to binary masks (need for Hausdorff calculation)
                preds_binary = (preds[i] > 0).astype(np.uint8)
                masks_binary = (masks[i] > 0).astype(np.uint8)

                # compute metrics only if both masks contain foreground objects
                if np.any(preds_binary) and np.any(masks_binary):
                    hausdorff_dist = hd(preds_binary, masks_binary)
                else:
                    hausdorff_dist = np.nan  # assign NaN if no foreground object is found

                dice_score = dc(preds_binary, masks_binary)  
                dice_scores.append(dice_score)
                hausdorff_dists.append(hausdorff_dist)

    # compute mean metrics for this fold
    avg_dice = np.nanmean(dice_scores)  # Use nanmean to ignore NaN values
    avg_hausdorff = np.nanmean(hausdorff_dists)
    fold_results.append((avg_dice, avg_hausdorff))

    print(f"Fold {fold+1}: Dice Score = {avg_dice:.4f}, Hausdorff Distance = {avg_hausdorff:.4f}")

# compute the final average results over all folds
avg_dice_final = np.nanmean([x[0] for x in fold_results])
avg_hausdorff_final = np.nanmean([x[1] for x in fold_results])

print("\nFinal Cross-Validation Results:")
print(f"Average Dice Score: {avg_dice_final:.4f}")
print(f"Average Hausdorff Distance: {avg_hausdorff_final:.4f}")
