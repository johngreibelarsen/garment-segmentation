import cv2
import glob
import matplotlib.pyplot as plt  
import skimage
from sklearn.metrics import f1_score
    
import predict_single_pass as predict_single_pass
import util.params as params

input_size = params.input_size
orig_width = params.orig_width
orig_height = params.orig_height
batch_size = params.batch_size
threshold = params.threshold


"""
Showing original image next to our prediction masks followed by DCIL/ISI cut-out 
mask and finally the effect of applying our mask to the original image
"""
def show_garment_images(original, prediction, mask_ground_truth):
    
    mask_predicted = prediction > threshold 
    #mask_ground_truth = cv2.cvtColor(mask_ground_truth, cv2.COLOR_BGR2GRAY)
    mask_ground_truth = mask_ground_truth > threshold
    
    print("\n***********************************")
    print("Dice f1: " + str(f1_score(mask_ground_truth.flatten(), mask_predicted.flatten(), average='binary')))
    print("***********************************")
       
    fig = plt.figure(figsize=(18, 18))
    
    imgplt = fig.add_subplot(1, 4, 1)
    imgplt.set_title("Original")
    plt.imshow(original)
    
    imgplt = fig.add_subplot(1, 4, 2)
    imgplt.set_title("Our Prediction")
    plt.imshow(mask_predicted)
 
    imgplt = fig.add_subplot(1, 4, 3)
    imgplt.set_title("ISI cut-out")
    plt.imshow(mask_ground_truth)

    imgplt = fig.add_subplot(1, 4, 4)
    imgplt.set_title("our auto cut-out")
    
    original = skimage.img_as_ubyte(original, force_copy=False)
    prediction = skimage.img_as_ubyte(mask_predicted, force_copy=False)
    our_cutout = cv2.bitwise_and(original, original, mask=prediction)  

    plt.imshow(our_cutout)
    
    plt.show()


"""
Demo showing original image next to our prediction masks followed by DCIL/ISI cut-out 
mask and finally the effect of applying our mask to the original image
"""
if __name__ == '__main__':
    img_orig_path = "./input/demonstration_set/original/*.*"
    img_mask_path = "./input/demonstration_set/mask/*.*"

    img_path_list = sorted(glob.glob(img_orig_path))
    mask_path_list = sorted(glob.glob(img_mask_path))
    
    for index in range(0, len(img_path_list)):
        img_orig = cv2.imread(img_path_list[index])
        predicted_mask = predict_single_pass.generate_mask(img_orig)
        ground_truth_mask = cv2.imread(mask_path_list[index], 0) # Reas as gray
        show_garment_images(img_orig, predicted_mask, ground_truth_mask)