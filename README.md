# Assignment-2-CAP5516

# Deep Learning-based Brain Tumor Segmentation Using MRI

This project performs **brain tumor segmentation** on MRI images using a **2D U-Net model**. The dataset used is a subset of the **Brain Tumor Image Segmentation (BraTS) challenge (2016-2017)**.

The implementation includes:
- **Data visualization** using ITK-SNAP and Matplotlib.
- **U-Net model training** with **5-fold cross-validation**.
- **Segmentation performance evaluation** using **Dice Score** and **Hausdorff Distance**.


## To Use:
(1) Visualizing MRI Scans
- To visualize the MRI scan and tumor in 3d download ITK-Snap from https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2 and choose Task01_BrainTumour.tar
  
  
- To visualize MRI images and their segmentation masks, run:
python mri_loader.py

(2) Training the U-Net Model
To train the model using 5-fold cross-validation, run:
python train_unet.py

(3) Testing the Model
To evaluate the trained model on test images and compute Dice Score & Hausdorff Distance, run:
python test_unet.py
