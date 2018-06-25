import cv2
import glob
import copy
import numpy as np
import matplotlib.pyplot as plt  
from skimage import exposure


def show_preprocessed_image(img_gray, img_color):    
    '''
    The following script illustrates the effect of histogram equalization and 
    contrast stretching techniques on gray and color images. Both the effect on 
    the image itself and the associated image histogram are displayed.
    '''
    
#----------------------- Gray stuff ------------------

    eq_hist_img = copy.copy(img_gray)
    eq_hist_img = cv2.equalizeHist(eq_hist_img)

    contrast_strech_img = copy.copy(img_gray)
    p10, p90 = np.percentile(contrast_strech_img, (10, 90))
    contrast_strech_img  = exposure.rescale_intensity(contrast_strech_img, in_range=(p10, p90))


    fig = plt.figure(figsize=(18, 18))

    imgplt = fig.add_subplot(1, 3, 1)
    imgplt.set_title("Un-processed image")
    plt.imshow(img_gray, cmap=plt.cm.bone)
    
    imgplt = fig.add_subplot(1, 3, 2)
    imgplt.set_title("Histogram equalization")
    plt.imshow(eq_hist_img, cmap=plt.cm.bone)

    imgplt = fig.add_subplot(1, 3, 3)
    imgplt.set_title("Contrast stretching")
    plt.imshow(contrast_strech_img, cmap=plt.cm.bone)

    fig = plt.figure(figsize=(18, 3))

    imgplt = fig.add_subplot(1, 3, 1)
    imgplt.set_title("Un-processed image")
    plt.hist(img_gray.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.grid(True)
    
    imgplt = fig.add_subplot(1, 3, 2)
    imgplt.set_title("Histogram equalization")
    plt.hist(eq_hist_img.flatten(),256,[0,256], color = 'g')
    plt.xlim([0,256])
    plt.grid(True)
    
    imgplt = fig.add_subplot(1, 3, 3)
    imgplt.set_title("Contrast stretching")
    plt.hist(contrast_strech_img.flatten(),256,[0,256], color = 'b')
    plt.xlim([0,256])
    plt.grid(True)

#------------------Color stuff -------------------------------

    eq_hist_color_img = copy.copy(img_color)
    img_y_cr_cb = cv2.cvtColor(eq_hist_color_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    y_eq = cv2.equalizeHist(y)
    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    eq_hist_color_img = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)    

    contrast_strech_color_img = copy.copy(img_color)
    p10, p90 = np.percentile(contrast_strech_color_img, (10, 90))
    contrast_strech_color_img  = exposure.rescale_intensity(contrast_strech_color_img, in_range=(p10, p90))


    figcolor = plt.figure(figsize=(18, 18))

    imgplt = figcolor.add_subplot(1, 3, 1)
    imgplt.set_title("Un-processed image")
    plt.imshow(img_color)
    
    imgplt = figcolor.add_subplot(1, 3, 2)
    imgplt.set_title("Histogram equalization")
    plt.imshow(eq_hist_color_img)

    imgplt = figcolor.add_subplot(1, 3, 3)
    imgplt.set_title("Contrast stretching")
    plt.imshow(contrast_strech_color_img)

    fig = plt.figure(figsize=(18, 3))

    imgplt = fig.add_subplot(1, 3, 1)
    imgplt.set_title("Un-processed image")
    plt.hist(img_color.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.grid(True)
    
    imgplt = fig.add_subplot(1, 3, 2)
    imgplt.set_title("Histogram equalization")
    plt.hist(eq_hist_color_img.flatten(),256,[0,256], color = 'g')
    plt.xlim([0,256])
    plt.grid(True)
    
    imgplt = fig.add_subplot(1, 3, 3)
    imgplt.set_title("Contrast stretching")
    plt.hist(contrast_strech_color_img.flatten(),256,[0,256], color = 'b')
    plt.xlim([0,256])
    plt.grid(True)    


''' 
Demo showing original image next to our contrast enhanced techniques: 
    - Histogram equalization
    - Contrast stretching    
'''
if __name__ == '__main__':
    img_orig_paths = "./input/demonstration_set/original/*.*"
    img_path = sorted(glob.glob(img_orig_paths))
    for file in img_path:
        img_color = cv2.imread(file)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        show_preprocessed_image(img_gray, img_color)
                