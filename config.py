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

# ====================================================
# DATA AUGMENTATION PARAMETERS
# ====================================================
def get_augmentations():
    """
    Returns two dictionaries: one for image augmentation and one for mask augmentation.
    They share most parameters, but masks typically require a different fill_mode and a constant value.
    """
    base_aug = {
        'rotation_range': 15,
        'width_shift_range': 0.05,
        'height_shift_range': 0.05,
        'shear_range': 0.05,
        'zoom_range': 0.1,
        'horizontal_flip': True,
    }
    img_aug = base_aug.copy()
    img_aug['vertical_flip'] = False
    img_aug['fill_mode'] = 'nearest'
    
    mask_aug = base_aug.copy()
    mask_aug['fill_mode'] = 'constant'
    mask_aug['cval'] = 0
    
    return img_aug, mask_aug

IMG_AUGMENTATION, MASK_AUGMENTATION = get_augmentations()

# ====================================================
# TRAINING PARAMETERS
# ====================================================
BATCH_SIZE = 16 # Change to 12 if there are errors regarding OOM.
NUM_EPOCHS = 10
TRAIN_TEST_SPLIT = 0.20
LEARNING_RATE = 1e-4
BACKBONE = 'resnet34'

# ====================================================
# FILE SAVING & OUTPUT PARAMETERS
# ====================================================
MODEL_SAVE_PATH_TEMPLATE = 'Resnet_model_Xe_2D_Vent_{}epochs.hdf5'
MATLAB_SAVE_X_TRAIN = 'Xe_X_train.mat'
MATLAB_SAVE_X_VAL = 'Xe_X_val.mat'
MATLAB_SAVE_Y_TRAIN = 'Xe_Y_train.mat'
MATLAB_SAVE_Y_VAL = 'Xe_Y_val.mat'

# New parameters for overlay visualization
SAVE_OVERLAY_IMAGES = True
OVERLAY_COLOR = (0, 255, 0)  # Green contour for overlays (OpenCV BGR format)