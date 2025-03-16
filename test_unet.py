import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from unet_model import UNet
from medpy.metric.binary import dc, hd #for Hausdorff distance

# load trained model
model = UNet(in_channels=1, out_channels=4).to(device)
model.load_state_dict(torch.load("unet_brain_tumor.pth", map_location=device))
model.eval()  # model set to evaluation 

# define test image and ground truth mask paths
test_image_path = "/Users/jojo/Task01_BrainTumour/imagesTr/BraTS_277.nii.gz"  # replace with your file location
test_mask_path = "/Users/jojo/Task01_BrainTumour/labelsTr/BraTS_277.nii.gz"  # replace with your file location

# load the MRI scan and segmentation mask
test_image_nifti = nib.load(test_image_path).get_fdata()
test_mask_nifti = nib.load(test_mask_path).get_fdata()

slice_idx = test_image_nifti.shape[2] // 2 #select slice
test_slice = test_image_nifti[:, :, slice_idx, 0]  # use FLAIR
test_mask = test_mask_nifti[:, :, slice_idx]  # mask for ground truth 

# normalize
test_slice = (test_slice - np.min(test_slice)) / (np.max(test_slice) - np.min(test_slice))

# convert to tensor
test_tensor = torch.tensor(test_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    pred_mask = model(test_tensor)
    pred_mask = torch.argmax(pred_mask, dim=1).squeeze().cpu().numpy()  # convert to numpy array

# compute Dice Score & Hausdorff Distance
preds_binary = (pred_mask > 0).astype(np.uint8)
masks_binary = (test_mask > 0).astype(np.uint8)

if np.any(preds_binary) and np.any(masks_binary):
    hausdorff_dist = hd(preds_binary, masks_binary)
else:
    hausdorff_dist = np.nan  # assign NaN if no foreground object is found

dice_score = dc(preds_binary, masks_binary)

print(f"Dice Score: {dice_score:.4f}")
print(f"Hausdorff Distance: {hausdorff_dist:.4f}")

# plot results, need space
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# for original MRI image
axes[0].imshow(test_slice, cmap="gray")
axes[0].set_title("MRI Scan (FLAIR)", fontsize=14, pad=15)  # increase font size & add padding
axes[0].axis("off")

# for corresponding predicted segmentation mask
axes[1].imshow(test_slice, cmap="gray")  # background MRI
axes[1].imshow(pred_mask, cmap="jet", alpha=0.5)  # overlay segmentation
axes[1].set_title("Predicted Segmentation Mask", fontsize=14, pad=15)
axes[1].axis("off")

# for corresponding ground truth segmentation mask
axes[2].imshow(test_slice, cmap="gray")
axes[2].imshow(test_mask, cmap="jet", alpha=0.5)  
axes[2].set_title("Ground Truth (GT) Mask", fontsize=14, pad=15)
axes[2].axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for titles
plt.savefig("test_unet_results.png", bbox_inches="tight")  # save with extra space
plt.show()

