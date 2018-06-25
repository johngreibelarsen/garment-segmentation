import os
import cv2
from tqdm import tqdm


def focus_garment_image(orig_img, mask_img):
    """
        Minimises the effect of noisy background from the original garment image by 
        first expanding (dilating) the garment mask with around 50 pixels in all directions 
        and considering everything outside that mask background that can be ignored.
        Returns image showing original image within dilated mask
    """            
    if (mask_img.shape[2] > 1): #if colour turn to gray
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    
    # Set threshold for what is garment and what is background
    ret, thresh = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY) 
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask_img = cv2.drawContours(mask_img, contours, -1, (255, 255, 255), 50)            
    img_no_BG = cv2.bitwise_and(orig_img, orig_img, mask=mask_img)        
    
    return img_no_BG
            
            
def save_focused_garment_list(orig_img_path, mask_path, img_no_background_path):
    """
        Minimises the effect of noisy background from the garment images by first 
        expanding (dilating) the garment masks with around 50 pixels in all directions 
        and considering everything outside that mask background that can be ignored.
        Writes image showing original image within dilated mask to specified path
    """
    for root, dirnames, filenames in os.walk(orig_img_path):
        for filename in tqdm(filenames):
            orig_img = cv2.imread(orig_img_path + filename)
            mask_img = cv2.imread(mask_path + filename)
            img_no_BG = focus_garment_image(orig_img, mask_img)                 
            cv2.imwrite(img_no_background_path + filename, img_no_BG)

                
# Demo of how background gets reduce to a minimum based on original images + Alpha masks
if __name__ == '__main__':
    orig_img_path = "../input/demonstration_set/original/"
    mask_path = "../input/demonstration_set/mask/"
    img_no_background_path = "../input/demonstration_set/original_no_background/"
    save_focused_garment_list(orig_img_path, mask_path, img_no_background_path)
    
    
