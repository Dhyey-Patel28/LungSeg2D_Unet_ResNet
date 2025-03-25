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
    
    if cfg.DEBUG_VALIDATION:
        print("DEBUG: Displaying a few random validation examples to check ground truth alignment and thresholding.")
        # Get one batch from the validation generator
        X_val_sample, Y_val_sample = val_generator[0]
        # Visualize a couple of examples using the helper function from HelpFunctions.py
        for _ in range(3):
            # Use the model (if already built) to run a debug sample.
            # To ensure the model is available, you can run this after model compilation,
            # or simply run it after training.
            pass  # We will call debug_validation_sample after model compilation
        
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
        classes=cfg.N_CLASSES,
        activation='sigmoid',
        decoder_use_batchnorm=True,
        decoder_filters=(256, 128, 64, 32, 16),
        encoder_freeze=True  # Freeze encoder layers initially
    )
    
    def dice_loss(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])  # Cast y_true to float32
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    # New weighted BCE loss function using parameters from config.py:
    def weighted_bce_loss(y_true, y_pred):
        # Create a weight tensor: lung pixels get higher weight
        weights = tf.where(tf.equal(y_true, 1), cfg.BCE_WEIGHT_LUNG, cfg.BCE_WEIGHT_BG)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weighted_bce = tf.reduce_mean(bce * weights)
        return weighted_bce
    
    model.compile(optimizer=Adam(learning_rate=cfg.LEARNING_RATE),
                loss=dice_loss,
                # metrics=[sm.metrics.iou_score]
                )
    print(model.summary())
    
    # If DEBUG_VALIDATION is enabled, run a debug sample:
    if cfg.DEBUG_VALIDATION:
        print("DEBUG: Running debug_validation_sample before training to inspect raw outputs.")
        HF.debug_validation_sample(val_generator, model)
    
    # -------------------- MODEL TRAINING --------------------
    # This configuration uses the F1-score (or Dice coefficient) as the primary metric for early stopping and checkpointing,
    # while still monitoring the validation loss for learning rate adjustments.
    callbacks = [
        # EarlyStopping: Stops training if the monitored metric ('val_f1-score') does not improve for a set number of epochs.
        #   - monitor='val_f1-score': Focus on segmentation quality (overlap between prediction and ground truth).
        #   - patience=20: Wait 20 epochs without improvement before stopping training.
        #   - mode='max': Since a higher F1-score is better, we look for a maximum.
        #   - Disadvantage: If the metric fluctuates, training might stop prematurely, especially if the improvements are subtle.
        EarlyStopping(monitor='val_f1-score', patience=20, mode='max', restore_best_weights=True),

        # ModelCheckpoint: Saves the model after every epoch if the monitored metric has improved.
        #   - monitor='val_f1-score', mode='max': Ensures we save the model with the best segmentation performance.
        #   - save_best_only=True: Saves only the best model, reducing storage overhead.
        #   - Disadvantage: If the metric is noisy, it might not update as frequently, and the 'best' model might be suboptimal.
        ModelCheckpoint(os.path.join(output_dir, 'best_model.h5'),
                        monitor='val_f1-score', mode='max', save_best_only=True),

        # ReduceLROnPlateau: Reduces the learning rate when the monitored metric (here, 'val_loss') has stopped improving.
        #   - monitor='val_loss': Uses validation loss as a signal for potential plateaus.
        #   - factor=0.5: Halves the learning rate each time the plateau condition is met.
        #   - patience=5: Waits 5 epochs before reducing the learning rate.
        #   - verbose=1: Prints a message when the learning rate is reduced.
        #   - Disadvantage: If the validation loss is noisy or the improvements are small, the LR may be reduced too quickly,
        #                   possibly slowing down training unnecessarily.
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]
    
    # Add class weight balancing
    class_weights = {0: cfg.BCE_WEIGHT_BG, 1: cfg.BCE_WEIGHT_LUNG}

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=cfg.NUM_EPOCHS,
        callbacks=callbacks,
        # class_weight=class_weights
    )
    
    # -------------------- SAVING TRAINING PLOTS & METRICS --------------------
    plots_dir = os.path.join(output_dir, "training_plots")
    HF.save_training_plots(history, output_dir=plots_dir)
    
    # -------------------- PREDICTION & MASK SAVING --------------------
    Y_pred_all = model.predict(val_generator, steps=len(val_generator))
    if Y_pred_all.shape[0] > 0:
        
        # Apply thresholding to convert soft probability outputs into binary masks
        Y_pred_all_thresh = (Y_pred_all > cfg.THRESHOLD).astype(np.uint8)
        if cfg.POSTPROCESS:
            # Process each slice in the volume (assuming shape is [num_slices, H, W])
            for i in range(Y_pred_all_thresh.shape[0]):
                Y_pred_all_thresh[i] = HF.post_process_mask(Y_pred_all_thresh[i])

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
    # Use configurable threshold for the batch predictions
    Y_pred_batch = (Y_pred_batch_probs > cfg.THRESHOLD).astype(np.uint8)
    
    if cfg.POSTPROCESS:
        for i in range(Y_pred_batch.shape[0]):
            slice_mask = Y_pred_batch[i, :, :, 0]
            processed = HF.post_process_mask(slice_mask)
            Y_pred_batch[i, :, :, 0] = np.squeeze(processed, axis=-1)
            
    Y_val_batch = Y_val_batch.astype(np.float32)  # Cast ground truth to float32

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
    
    # Run debug again:
    if cfg.DEBUG_VALIDATION:
        print("DEBUG: Running debug_validation_sample after training to inspect model outputs.")
        HF.debug_validation_sample(val_generator, model)
    
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
        # Use the configurable threshold instead of hard-coded 0.5
        Y_pred_batch = (Y_pred_batch_probs > cfg.THRESHOLD).astype(np.uint8)
        
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

    # Print a single message after all batches have been processed
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
