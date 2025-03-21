import os
os.environ["SM_FRAMEWORK"] = "tf.keras"  # Must be set before any other imports
import random
import time
import csv
import datetime
import cv2  # NEW: For overlay visualization
 
# Third‑Party Libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import nibabel as nib
from scipy.io import savemat
from sklearn.model_selection import train_test_split
import seaborn as sns

try:
    from medpy.metric.binary import hd
    USE_MEDPY = True
except ImportError:
    print("medpy not installed; Hausdorff distance will be skipped.")
    USE_MEDPY = False

# TensorFlow / Keras Imports
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # NEW: Import scheduler
from keras.optimizers import Adam
from keras import utils as keras_utils
if not hasattr(keras_utils, "generic_utils"):
    keras_utils.generic_utils = type("dummy", (), {"get_custom_objects": keras_utils.get_custom_objects})

# Segmentation Models
import segmentation_models as sm

# Local Modules
import HelpFunctions as HF  # Contains SliceSequence and helper functions
from matplotlib.widgets import Slider  # For interactive visualization

# Import configuration
import config as cfg

# ====================================================
# TRAINING JOB FUNCTION
# ====================================================
def training_job():
    print("Starting training job...")

    # Create a unique output directory based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.OUTPUT_BASE_DIR, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------- DATA GENERATION --------------------
    # Build list of subject directories from DATA_PATH.
    # Each subject directory is assumed to now contain saved slice files in 'proton' and 'mask' subfolders.
    subject_dirs = [os.path.join(cfg.DATA_PATH, d) for d in os.listdir(cfg.DATA_PATH)
                    if os.path.isdir(os.path.join(cfg.DATA_PATH, d))]
    print("Total subjects found:", len(subject_dirs))
    
    # Split subject directories into training and validation sets.
    train_dirs, val_dirs = train_test_split(subject_dirs, test_size=cfg.TRAIN_TEST_SPLIT, random_state=42)
    print("Train subjects:", len(train_dirs))
    print("Validation subjects:", len(val_dirs))
    
    # -------------------- GENERATORS --------------------
    # Use the NiftiSliceSequence data generator which loads individual slice files.
    train_generator = HF.NiftiSliceSequence(
        slice_dirs=train_dirs,
        batch_size=cfg.BATCH_SIZE,
        image_size=cfg.IMAGE_SIZE,
        img_aug=cfg.IMG_AUGMENTATION,
        mask_aug=cfg.MASK_AUGMENTATION,
        augment=True,
        shuffle=True,
        max_slices_per_subject=50  # optional: limit number of slices per subject
    )
    
    val_generator = HF.NiftiSliceSequence(
        slice_dirs=val_dirs,
        batch_size=cfg.BATCH_SIZE,
        image_size=cfg.IMAGE_SIZE,
        img_aug=cfg.IMG_AUGMENTATION,
        mask_aug=cfg.MASK_AUGMENTATION,
        augment=False,
        shuffle=False
    )
        
    # Optionally visualize a few augmented samples from the training generator.
    HF.visualize_augmented_samples_overlay(train_generator, num_samples=5)
    HF.visualize_augmented_samples(train_generator, num_samples=2)
    
    # -------------------- MODEL DEFINITION --------------------
    BACKBONE = cfg.BACKBONE
    preprocess_input = sm.get_preprocessing(BACKBONE)
    model = sm.Unet(
        BACKBONE,
        encoder_weights='imagenet',
        input_shape=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3),
        classes=1,
        activation='sigmoid'
    )
    
    def dice_loss(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    def bce_dice_loss(y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
        dsc = dice_loss(y_true, y_pred)
        return bce + dsc
    
    model.compile(optimizer=Adam(learning_rate=cfg.LEARNING_RATE),
                  loss=bce_dice_loss,
                  metrics=[sm.metrics.iou_score])
    print(model.summary())
    
    # -------------------- MODEL TRAINING --------------------
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ModelCheckpoint(os.path.join(output_dir, 'best_model.keras'),
                        monitor='val_loss', save_best_only=True),  # Added comma here
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)  # NEW: LR scheduler
    ]
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=cfg.NUM_EPOCHS,
        callbacks=callbacks
    )
    
    # -------------------- SAVING TRAINING PLOTS & METRICS --------------------
    plots_dir = os.path.join(output_dir, "training_plots")
    HF.save_training_plots(history, output_dir=plots_dir)
    
    # -------------------- PREDICTION & MASK SAVING --------------------
    Y_pred_all = model.predict(val_generator, steps=len(val_generator))
    if Y_pred_all.shape[0] > 0:
        # Apply thresholding to convert soft probability outputs into binary masks
        Y_pred_all_thresh = (Y_pred_all > 0.5).astype(np.uint8)
        # Convert the thresholded predictions into a volume:
        pred_volume = np.transpose(Y_pred_all_thresh.squeeze(-1), (1, 2, 0))
        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        HF.save_masks(
            pred_volume,
            mat_path=os.path.join(output_dir, "final_gen_mask.mat"),
            nifti_path=os.path.join(output_dir, "final_gen_mask.nii.gz"),
            png_dir=masks_dir,
            meta_info=val_generator.get_meta_info()  # Call the method instead of accessing a non-existent property
        )
    else:
        print("No predictions were made; skipping mask saving.")

    
    # -------------------- EVALUATION & VISUALIZATION ON ONE BATCH --------------------
    batch_idx = random.randint(0, len(val_generator) - 1)
    X_val_batch, Y_val_batch = val_generator[batch_idx]
    Y_pred_batch_probs = model.predict(X_val_batch)
    # Immediately threshold the predictions:
    Y_pred_batch = (Y_pred_batch_probs > 0.5).astype(np.uint8)

    HF.plot_validation_dice(Y_val_batch, Y_pred_batch, output_dir=plots_dir)

    HF.evaluate_and_save_segmentation_plots(
        Y_true=Y_val_batch,
        Y_pred_probs=Y_pred_batch_probs,
        Y_pred_bin=Y_pred_batch,
        output_dir=plots_dir,
        prefix="val"
    )
    
    sample_val_idx = random.randint(0, X_val_batch.shape[0] - 1)
    HF.visualize_image_and_mask(
        X_val_batch[sample_val_idx][:, :, 0],
        Y_val_batch[sample_val_idx][:, :, 0],
        title="Ground Truth"
    )
    HF.visualize_image_and_mask(
        X_val_batch[sample_val_idx][:, :, 0],
        Y_pred_batch[sample_val_idx][:, :, 0],
        title="Prediction"
    )
    
    # -------------------- NEW: OVERLAY VISUALIZATION --------------------
    if cfg.SAVE_OVERLAY_IMAGES:
        # Convert original image to uint8 format for contour detection
        original_image = (X_val_batch[sample_val_idx][:, :, 0] * 255).astype(np.uint8)
        # Ensure binary mask is in uint8 format
        mask = (Y_pred_batch[sample_val_idx][:, :, 0] * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(overlay_image, contours, -1, (0, 255, 0), 2)
        overlay_save_path = os.path.join(plots_dir, "prediction_overlay.png")
        cv2.imwrite(overlay_save_path, overlay_image)
        print("Saved overlay image to", overlay_save_path)
        
    # This is a new block to generate overlays for ALL slices in the validation set
    all_comparisons_dir = os.path.join(output_dir, "all_comparisons")
    os.makedirs(all_comparisons_dir, exist_ok=True)
    
    # Loop over every batch in the validation generator
    for batch_idx in range(len(val_generator)):
        X_val_batch, Y_val_batch = val_generator[batch_idx]
        Y_pred_batch_probs = model.predict(X_val_batch)
        Y_pred_batch = (Y_pred_batch_probs > 0.5).astype(np.uint8)
        
        # Loop over every slice in this batch
        for slice_idx in range(X_val_batch.shape[0]):
            # Extract raw data, ground truth, and prediction
            raw_image = X_val_batch[slice_idx, :, :, 0]     # single-channel slice
            ground_truth = Y_val_batch[slice_idx, :, :, 0]
            prediction = Y_pred_batch[slice_idx, :, :, 0]
            
            # Create side-by-side overlay
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
            ax[0].imshow(raw_image, cmap='gray')
            ax[0].imshow(ground_truth, cmap='Reds', alpha=0.3)
            ax[0].set_title("Ground Truth Overlay")
            ax[0].axis('off')
            
            ax[1].imshow(raw_image, cmap='gray')
            ax[1].imshow(prediction, cmap='Blues', alpha=0.3)
            ax[1].set_title("Prediction Overlay")
            ax[1].axis('off')
            
            # Save to a file named by batch_idx and slice_idx
            comparison_filename = f"comparison_batch{batch_idx}_slice{slice_idx:03d}.png"
            comparison_path = os.path.join(all_comparisons_dir, comparison_filename)
            plt.savefig(comparison_path, dpi=300)
            plt.close()
    
    print("Saved comparison overlays for all slices in validation set.")
            
    # -------------------- SAVE MODEL --------------------
    model_save_path = os.path.join(output_dir, cfg.MODEL_SAVE_PATH_TEMPLATE.format(cfg.NUM_EPOCHS))
    model.save(model_save_path)
    
    print("Training job completed. Outputs saved in:", output_dir)

    # Assume you are using the validation generator (with shuffle disabled) so that order is maintained.
    meta_info = val_generator.get_meta_info()  # Pass the metadata list

    csv_file_path = os.path.join(output_dir, "mask_metadata.csv")
    with open(csv_file_path, "w", newline="") as csvfile:
        fieldnames = ["patient_id", "slice_number", "image_file", "mask_file"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # We assume that the saved individual slice PNG files are in the "masks" folder and are named as "slice_{i+1:03}.png"
        for i, meta in enumerate(meta_info):
            mask_filename = os.path.join("masks", f"slice_{i+1:03}.png")
            writer.writerow({
                "patient_id": meta["patient_id"],
                "slice_number": meta["slice_number"],
                "image_file": meta["image_path"],
                "mask_file": mask_filename
            })

    print("Saved metadata CSV to", csv_file_path)

def main():
    start_time = time.time()
    training_job()
    elapsed = time.time() - start_time
    print(f"Training job finished in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
