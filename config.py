import os
import numpy as np

# Create a folder named 'LungSegmentation2D_Unet_BBResnet_Outputs' in the parent directory of this file
OUTPUT_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'LungSegmentation2D_Unet_BBResnet_Outputs')

# ====================================================
# DATA LOADING & PREPROCESSING PARAMETERS
# ====================================================
DATA_PATH = 'Neonatal Test Data - August 1 2024'
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_PATH) 
IMAGE_SIZE = 256  # assumed square images
CHANNELS = 1  # Changed to single channel input
N_CLASSES = 1

# ====================================================
# DATA AUGMENTATION PARAMETERS
# ====================================================
def get_augmentations():
    base_aug = {
        'rotation_range': 10,
        'width_shift_range': 0.05,
        'height_shift_range': 0.05,
        'shear_range': 0.05,
        'zoom_range': 0.15,
        'horizontal_flip': True,
        'vertical_flip': False,
        'fill_mode': 'constant'
    }
    
    img_aug = base_aug.copy()
    img_aug['fill_mode'] = 'constant'
    img_aug['cval'] = 0.0
    
    mask_aug = base_aug.copy()
    mask_aug['fill_mode'] = 'constant'
    mask_aug['cval'] = 0
    
    return img_aug, mask_aug

IMG_AUGMENTATION, MASK_AUGMENTATION = get_augmentations()

# ====================================================
# TRAINING PARAMETERS
# ====================================================
BATCH_SIZE = 32  # Increased batch size
NUM_EPOCHS = 1  # Increased epochs
TRAIN_TEST_SPLIT = 0.15
LEARNING_RATE = 1e-4  # Reduced learning rate

# ====================================================
# BACKBONE
# ====================================================
'''
resnet18:
  - A shallow network with fewer parameters.
  - Provides faster inference and requires less memory.
  - May be sufficient for simpler tasks or when computational resources are very limited,
    but might not capture complex features as well as deeper networks.

resnet34:
  - Slightly deeper than resnet18 and offers a better feature representation.
  - Still relatively efficient, balancing speed and accuracy.

resnet50:
  - A deeper and more robust network with a higher capacity for feature extraction.
  - Pre-trained on ImageNet, often yielding better performance on complex tasks.
  - However, it is computationally heavier compared to resnet18/34.

efficientnetb0:
  - Highly optimized for both accuracy and efficiency using compound scaling.
  - Typically offers a great trade-off between performance and computational cost,
    making it a strong candidate for segmentation tasks with limited data.

efficientnetb1:
  - Slightly larger than efficientnetb0 with a potential boost in accuracy.
  - Comes with increased computation and parameter count.

Additional backbone options that can be considered for lung segmentation:

mobilenetv2:
  - Extremely lightweight and fast.
  - Ideal for edge or mobile applications, though it may require extra tuning for high accuracy.

densenet121:
  - Uses dense connectivity for better feature propagation.
  - Often effective for segmentation tasks, but typically uses more memory.

inceptionv3:
  - Provides multi-scale feature extraction.
  - Less common in segmentation pipelines but can be effective in capturing varied features.
'''
# BACKBONE = 'resnet18'        # Fast and lightweight but may miss complex features.
# BACKBONE = 'resnet34'        # A good balance of efficiency and feature representation.
BACKBONE = 'resnet50'         # Strong feature extractor, robust but computationally heavier.
# BACKBONE = 'efficientnetb0'   # Excellent trade-off between performance and efficiency.
# BACKBONE = 'efficientnetb1'   # Slightly larger than b0 for potential accuracy gains.
# BACKBONE = 'mobilenetv2'      # Very lightweight; ideal for resource-constrained scenarios.
# BACKBONE = 'densenet121'      # Dense connections aid feature propagation but increase memory usage.
# BACKBONE = 'inceptionv3'      # Multi-scale feature extraction, though less common for segmentation.


# ====================================================
# CLASS IMBALANCE PARAMETERS & REGULARIZATION PARAMETERS
# ====================================================
BCE_WEIGHT_LUNG = 10.0   # Weight for lung (foreground) pixels in BCE
BCE_WEIGHT_BG = 1.0      # Weight for background pixels in BCE
DICE_WEIGHT = 1.5        # Weight for Dice loss component
DROPOUT_RATE = 0.2       # Dropout rate to help prevent overfitting
WEIGHT_DECAY = 1e-5      # Weight decay (L2 regularization)

# ====================================================
# THRESHOLDING & POST-PROCESSING PARAMETERS
# ====================================================
THRESHOLD = 0.1          # Use a lower threshold to capture faint lung regions
POSTPROCESS = False      # Whether to apply post-processing to predicted masks
MORPH_KERNEL_SIZE = 5    # Kernel size for morphological closing

# ====================================================
# DEBUGGING PARAMETERS
# ====================================================
DEBUG_VALIDATION = True  # Set to True to visualize random validation samples

# ====================================================
# FILE SAVING & OUTPUT PARAMETERS
# ====================================================
MODEL_SAVE_PATH_TEMPLATE = f"{BACKBONE[0].upper() + BACKBONE[1:]}__model_Xe_2D_Vent_{{}}epochs.hdf5"
MATLAB_SAVE_X_TRAIN = 'Xe_X_train.mat'
MATLAB_SAVE_X_VAL = 'Xe_X_val.mat'
MATLAB_SAVE_Y_TRAIN = 'Xe_Y_train.mat'
MATLAB_SAVE_Y_VAL = 'Xe_Y_val.mat'

# New parameters for overlay visualization
SAVE_OVERLAY_IMAGES = True
OVERLAY_COLOR = (0, 255, 0)  # Green contour for overlays (OpenCV BGR format)
