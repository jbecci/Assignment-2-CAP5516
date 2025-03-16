# Assignment-2-CAP5516

# Deep Learning-based Brain Tumor Segmentation Using MRI

This project performs **brain tumor segmentation** on MRI images using a **2D U-Net model**. 

The implementation includes:
- **Data visualization** using ITK-SNAP.
- **U-Net model training** with **5-fold cross-validation**.
- **Performance evaluation** using **Dice Score** & **Hausdorff Distance**.


## To Use:
(1) Visualizing MRI Scans
- To visualize the MRI scan and tumor in 3d, download ITK-SNAP from http://www.itksnap.org/pmwiki/pmwiki.php
- Download BraTS dataset from https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2 and choose Task01_BrainTumour.tar.
- Load one BraTS MRI and corresponding segmentation mask into ITK-SNAP

(2) Initialize 2D U-Net Model
To initialize the model before training, run:
`python unet_model.py`

(3) Training the U-Net Model
To train the model using 5-fold cross-validation, run:
`python train_unet.py`

(4) Testing the Model
To evaluate the trained model on test images and compute Dice Score & Hausdorff Distance, run:
`python test_unet.py`
