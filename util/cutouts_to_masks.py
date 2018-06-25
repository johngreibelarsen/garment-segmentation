import cv2
import os
from tqdm import tqdm
import skimage
import numpy as np
from matplotlib import pyplot as plt

import util.params as params

orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold

def turn_image_to_mask(cutout_image):
    """
        Estracts the Alpha mask from the supplied image. It is assumed that the 
        image format follows: (height, width, channels), ex: (1728, 1152, 4)
    """
    if cutout_image.shape[2] != 4:
        raise ValueError("image format must follows(height, width, channels) with channels = 4 (RGBA)")
    img_reshaped = np.moveaxis(cutout_image, 2, 0)
    mask = img_reshaped[3] # index of the alpha mask
    return mask


def cutouts_to_masks(cutout_path, save_mask_path):
    """
        Estracts the Alpha masks from the supplied images contained in cutout_path location and saves 
        the masks to save_mask_path location. It is assumed that the image format follows: 
        (height, width, channels), ex: (1728, 1152, 4)
    """
    for root, dirnames, filenames in os.walk(cutout_path):
        for filename in tqdm(filenames):
            cutout_image = cv2.imread(cutout_path + filename, cv2.IMREAD_UNCHANGED)
            cutout_mask = turn_image_to_mask(cutout_image)
            cv2.imwrite(save_mask_path + filename, cutout_mask)
            
            
def show_cutouts_and_masks(orig_img_path, cutout_paths):
    """
        For a given set of original images and cut-out locations (paths)
        calculates the Alpha mask and displays the original image, the calculated mask and
        the the calculated cut-out image (using original and calculated mask)
    """
    for root, dirnames, filenames in os.walk(cutout_paths):
        for filename in tqdm(filenames):
            orig_image = cv2.imread(orig_img_path + filename)
            cutout_image = cv2.imread(cutout_path + filename, cv2.IMREAD_UNCHANGED)
            cutout_mask = turn_image_to_mask(cutout_image)
            
            fig = plt.figure(figsize=(20, 20))
        
            imgplt = fig.add_subplot(1, 3, 1)
            imgplt.set_title("Original")
            plt.imshow(orig_image)
            
            imgplt = fig.add_subplot(1, 3, 2)
            imgplt.set_title("Mask")
            plt.imshow(cutout_mask)

            orig = skimage.img_as_ubyte(orig_image, force_copy=False)
            mask = skimage.img_as_ubyte(cutout_mask, force_copy=False)
            the_cutout = cv2.bitwise_and(orig, orig, mask=mask)  

            imgplt = fig.add_subplot(1, 3, 3)
            imgplt.set_title("Cutout")
            plt.imshow(the_cutout)
        
# Demo of how DCIL or ISI cutouts are turned into binary masks
if __name__ == '__main__':
    orig_img_path = "../input/demonstration_set/original/"
    cutout_path = "../input/demonstration_set/cutout/"
    show_cutouts_and_masks(orig_img_path, cutout_path)
    
    #save_mask_path = "../input/demonstration_set/mask/"    
    #cutouts_to_masks(cutout_path, save_mask_path)