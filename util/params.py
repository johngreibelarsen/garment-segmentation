"""
    Our config file for training and predicting our U-net model
"""

from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024

# Image sizes
orig_width = 1152 # original width of image
orig_height = 1728 # original height of image


# Model parameters
model_lr=3e-4 # The default learning rate that the a selected model is born with
input_size = 1024 # The image input size to the model: 128, 256, 512 or 1024
max_epochs = 100 # Max epochs to use for training
batch_size = 3 # Batch size to use for reading and processing images
model_factory = get_unet_1024 # The model choosen


# Learning rate parameters (for use with cyclic_learning_rate.py)
plateau_steps = 1 # The initial no. of epocs were the LR will be constant at (min_lr + max_lr)/2
step_size = 10 # 1/2 the wave length of the oscillating LR function
min_lr = 7e-5 # Min lR
max_lr = 7e-4 # Max LR


# Mask cut-out parameters
threshold = 0.45 # the probability threshold above which a pixel is decided as part of the image mask
threshold_when_focused = 0.15 # the probability threshold when using a focused/dilated mask

# Centering and normalizing inout data matrixes
centering_mean_path = './normalization/unet_no_focus_mask_1024_meanstd_resize_optimized_mean.npy'
normalizing_std_path = './normalization/unet_no_focus_mask_1024_meanstd_resize_optimized_std.npy'

# Model weights to use for predictions
weight_no_focus_path = './weights/unet_no_focus_mask_1024_meanstd_resize_optimized.hdf5'
weight_with_focus_path = './weights/unet_with_focus_mask_1024_not_optimized.hdf5' # Image through dilated mask
