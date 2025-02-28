import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import nibabel as nib
from scipy.io import savemat
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from tensorflow.keras.utils import Sequence
import tensorflow as tf  # Needed for tf.keras.preprocessing.image.ImageDataGenerator

try:
    from medpy.metric.binary import hd
    USE_MEDPY = True
except ImportError:
    print("medpy not installed; Hausdorff distance will be skipped.")
    USE_MEDPY = False

# ---------------------------------------------------------------------
# Function to save individual slices from a 3D volume
# ---------------------------------------------------------------------
def save_individual_slices(subject_dir, output_dir, image_size, max_slices=16):
    """
    Opens the proton and mask NIfTI volumes from subject_dir,
    resizes them to (image_size x image_size), and saves individual slice images.
    
    The output directory will be organized as:
      output_dir/
          proton/   -> saved proton slice PNGs
          mask/     -> saved mask slice PNGs
    Only up to max_slices slices are saved per subject.
    """
    os.makedirs(output_dir, exist_ok=True)
    proton_output = os.path.join(output_dir, 'proton')
    mask_output = os.path.join(output_dir, 'mask')
    os.makedirs(proton_output, exist_ok=True)
    os.makedirs(mask_output, exist_ok=True)
    
    proton_file = _find_proton_file(subject_dir)
    mask_file = _find_mask_file(subject_dir)
    if proton_file is None or mask_file is None:
        print(f"Skipping {subject_dir}: missing proton or mask file.")
        return
    
    # Load volumes and convert to float32
    proton_data = nib.load(proton_file).get_fdata().astype(np.float32)
    mask_data = nib.load(mask_file).get_fdata().astype(np.float32)
    
    # Resize volumes (the third dimension—number of slices—is kept as is)
    proton_data = resize(proton_data, (image_size, image_size, proton_data.shape[2]),
                         mode='constant', preserve_range=True, order=1)
    mask_data = resize(mask_data, (image_size, image_size, mask_data.shape[2]),
                       mode='constant', preserve_range=True, order=0)
    
    # Use the minimum number of slices between proton and mask volumes
    num_slices = min(proton_data.shape[2], mask_data.shape[2])
    if num_slices > max_slices:
        slice_indices = np.sort(np.random.choice(num_slices, max_slices, replace=False))
    else:
        slice_indices = np.arange(num_slices)
    
    for i in slice_indices:
        proton_slice = proton_data[:, :, i]
        mask_slice = (mask_data[:, :, i] > 0).astype(np.uint8)
        proton_path = os.path.join(proton_output, f"proton_slice_{i:03d}.png")
        mask_path = os.path.join(mask_output, f"mask_slice_{i:03d}.png")
        plt.imsave(proton_path, proton_slice, cmap='gray')
        plt.imsave(mask_path, mask_slice, cmap='gray')
    print(f"Saved slices for subject {subject_dir} to {output_dir}")

def _find_proton_file(subject_dir):
    candidates = glob.glob(os.path.join(subject_dir, '*[Pp]roton*.*nii*'))
    return candidates[0] if candidates else None

def _find_mask_file(subject_dir):
    candidates = glob.glob(os.path.join(subject_dir, '*[Mm]ask*.*nii*'))
    for candidate in candidates:
        if os.path.basename(candidate).lower() == "mask.nii":
            return candidate
    return candidates[0] if candidates else None

# ---------------------------------------------------------------------
# Data Generator for loading individual slice files using Keras Sequence
# ---------------------------------------------------------------------
class SliceSequence(Sequence):
    def __init__(self, slice_dirs, batch_size, image_size, 
                 img_aug=None, mask_aug=None, augment=False, shuffle=True):
        """
        slice_dirs: list of directories, one per subject, where each directory contains
                    two subfolders: 'proton' and 'mask' with individual slice PNG files.
        batch_size: number of slice pairs per batch.
        image_size: target image size (assumed square). Slices are loaded and resized if needed.
        img_aug: dictionary of image augmentation parameters.
        mask_aug: dictionary of mask augmentation parameters.
        augment: Boolean flag to apply augmentation.
        shuffle: whether to shuffle the slice file indices each epoch.
        """
        self.slice_dirs = slice_dirs
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.shuffle = shuffle
        
        # Build lists of slice file paths across all provided subject directories.
        self.image_files = []
        self.mask_files = []
        for d in self.slice_dirs:
            proton_dir = os.path.join(d, 'proton')
            mask_dir = os.path.join(d, 'mask')
            proton_files = sorted(glob.glob(os.path.join(proton_dir, '*.png')))
            mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
            if len(proton_files) == len(mask_files):
                self.image_files.extend(proton_files)
                self.mask_files.extend(mask_files)
            else:
                print(f"Warning: Mismatch in number of slices in {d}")
        self.indexes = np.arange(len(self.image_files))
        self.on_epoch_end()
        
        if self.augment:
            self.img_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**(img_aug if img_aug else {}))
            self.mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**(mask_aug if mask_aug else {}))
    
    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_img_files = [self.image_files[i] for i in batch_indexes]
        batch_mask_files = [self.mask_files[i] for i in batch_indexes]
        X, Y = self.__data_generation(batch_img_files, batch_mask_files)
        return X, Y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, batch_img_files, batch_mask_files):
        X_batch = []
        Y_batch = []
        
        for img_file, mask_file in zip(batch_img_files, batch_mask_files):
            img = plt.imread(img_file)
            mask = plt.imread(mask_file)
            
            # Ensure images have a channel dimension
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=-1)
                
            # Resize if necessary
            if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                img = resize(img, (self.image_size, self.image_size),
                            mode='constant', preserve_range=True)
                mask = resize(mask, (self.image_size, self.image_size),
                            mode='constant', preserve_range=True, order=0)
                
            if self.augment:
                transform = self.img_datagen.get_random_transform(img.shape)
                img = self.img_datagen.apply_transform(img, transform)
                # Force nearest-neighbor interpolation for the mask
                mask = self.mask_datagen.apply_transform(mask, transform, order=0)
            
            X_batch.append(img)
            Y_batch.append(mask)
        
        X_batch = np.stack(X_batch, axis=0)
        Y_batch = np.stack(Y_batch, axis=0)
        
        # If images are single-channel, replicate to 3 channels.
        if X_batch.shape[-1] == 1:
            X_batch = np.repeat(X_batch, 3, axis=-1)
            
        X_batch = X_batch.astype('float32') / 255.0
        Y_batch = Y_batch.astype('float32')
        
        return X_batch, Y_batch

# ---------------------------------------------------------------------
# Other Utility Functions (unchanged)
# ---------------------------------------------------------------------
def save_masks(final_mask, mat_path, nifti_path, png_dir):
    # Apply thresholding to ensure binary masks
    final_mask = (final_mask > 0.5).astype(np.uint8)
    savemat(mat_path, {"final_gen_mask": final_mask})
    print(f"Generated mask saved as {mat_path}.")
    nifti_img = nib.Nifti1Image(final_mask, affine=np.eye(4))
    nib.save(nifti_img, nifti_path)
    print(f"Generated mask saved as {nifti_path}.")
    os.makedirs(png_dir, exist_ok=True)
    for i in range(final_mask.shape[2]):
        plt.imsave(os.path.join(png_dir, f"slice_{i+1:03}.png"), final_mask[:, :, i], cmap='gray')
    print(f"Predicted slices saved as PNGs in {png_dir}.")


def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure()
    plt.plot(epochs, history.history['loss'], 'y', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_validation_loss.png'), dpi=300)
    plt.close()
    plt.figure()
    plt.plot(epochs, history.history['iou_score'], 'y', label='Training IoU')
    plt.plot(epochs, history.history['val_iou_score'], 'r', label='Validation IoU')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU Score')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_validation_iou.png'), dpi=300)
    plt.close()

def visualize_image_and_mask(image, mask, title="Image and Mask"):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"{title} - Image")
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title(f"{title} - Mask")
    plt.imshow(mask, cmap='gray')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def visualize_slice(image_volume, mask_volume, slice_idx, title_prefix=""):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"{title_prefix} - Image (Slice {slice_idx})")
    plt.imshow(image_volume[:, :, slice_idx], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title(f"{title_prefix} - Mask (Slice {slice_idx})")
    plt.imshow(mask_volume[:, :, slice_idx], cmap='gray')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def visualize_augmented_samples(generator, num_samples=3):
    for _ in range(num_samples):
        idx = np.random.randint(0, len(generator))
        img_batch, mask_batch = generator[idx]
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Augmented Image")
        plt.imshow(img_batch[0, :, :, 0], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Augmented Mask")
        plt.imshow(mask_batch[0, :, :, 0], cmap='gray')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

def visualize_augmented_samples_overlay(generator, num_samples=3):
    for _ in range(num_samples):
        idx = np.random.randint(0, len(generator))
        img_batch, mask_batch = generator[idx]
        image = img_batch[0]  # (H, W, 3)
        mask = mask_batch[0, :, :, 0]  # (H, W)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Augmented Image")
        plt.imshow(image[..., 0], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Mask Overlay")
        plt.imshow(image[..., 0], cmap='gray')
        plt.imshow(mask, alpha=0.3, cmap='Reds')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

def plot_validation_dice(y_true, y_pred, output_dir):
    def dice_coef_np(y_true, y_pred, smooth=1e-6):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    dice_value = dice_coef_np(y_true, (y_pred > 0.5).astype(np.float32))
    plt.figure()
    plt.bar(['Dice Coefficient'], [dice_value], color='skyblue')
    plt.title('Validation Dice Coefficient')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'validation_dice.png'), dpi=300)
    plt.close()
    print("Validation Dice coefficient:", dice_value)

def evaluate_and_save_segmentation_plots(Y_true, Y_pred_probs, Y_pred_bin, output_dir, prefix="val"):
    os.makedirs(output_dir, exist_ok=True)
    if Y_true.ndim == 4:
        Y_true = Y_true[..., 0]
    if Y_pred_probs.ndim == 4:
        Y_pred_probs = Y_pred_probs[..., 0]
    if Y_pred_bin.ndim == 4:
        Y_pred_bin = Y_pred_bin[..., 0]
    def dice_coef_np(y_true, y_pred, smooth=1e-6):
        intersection = np.sum(y_true * y_pred)
        return (2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    dice_per_slice = []
    for i in range(Y_true.shape[0]):
        dsc = dice_coef_np(Y_true[i], Y_pred_bin[i])
        dice_per_slice.append(dsc)
    plt.figure(figsize=(6, 5))
    plt.boxplot(dice_per_slice)
    plt.title("Dice Coefficient Distribution")
    plt.ylabel("DSC")
    plt.ylim([0, 1])
    plt.savefig(os.path.join(output_dir, f"{prefix}_dice_boxplot.png"), dpi=300)
    plt.close()
    
    y_true_flat = Y_true.flatten()
    y_prob_flat = Y_pred_probs.flatten()
    fpr, tpr, _ = roc_curve(y_true_flat, y_prob_flat)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f"{prefix}_ROC_curve.png"), dpi=300)
    plt.close()
    
    y_pred_flat = Y_pred_bin.flatten().astype(int)
    cm = confusion_matrix(Y_true.flatten(), y_pred_flat, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"), dpi=300)
    plt.close()
    
    if USE_MEDPY:
        from medpy.metric.binary import hd
        hausdorff_list = []
        for i in range(Y_true.shape[0]):
            gt = (Y_true[i] > 0.5).astype(np.uint8)
            pr = (Y_pred_bin[i] > 0.5).astype(np.uint8)
            if gt.sum() == 0 and pr.sum() == 0:
                hausdorff_list.append(0.0)
                continue
            try:
                hdist = hd(pr, gt)
                hausdorff_list.append(hdist)
            except:
                hausdorff_list.append(np.nan)
        valid_hds = [h for h in hausdorff_list if not np.isnan(h)]
        if len(valid_hds) > 0:
            plt.figure(figsize=(6, 5))
            plt.boxplot(valid_hds)
            plt.title("Hausdorff Distance Distribution")
            plt.ylabel("Hausdorff distance (pixels)")
            plt.savefig(os.path.join(output_dir, f"{prefix}_hausdorff_boxplot.png"), dpi=300)
            plt.close()
    else:
        print("Hausdorff distance not computed because medpy is not installed.")
    
    n_examples = 5
    random_indices = np.random.choice(range(Y_true.shape[0]), size=n_examples, replace=False)
    overlay_dir = os.path.join(output_dir, f"{prefix}_overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    for i in random_indices:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].set_title("Ground Truth Overlay")
        ax[0].imshow(Y_pred_probs[i], cmap='gray')
        ax[0].imshow(Y_true[i], alpha=0.3, cmap='Reds')
        ax[0].axis('off')
        ax[1].set_title("Prediction Overlay")
        ax[1].imshow(Y_pred_probs[i], cmap='gray')
        ax[1].imshow(Y_pred_bin[i], alpha=0.3, cmap='Reds')
        ax[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(overlay_dir, f"overlay_{i:03d}.png"), dpi=200)
        plt.close()
