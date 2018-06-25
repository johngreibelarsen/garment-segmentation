"""
Utility script for calculating Dice scores for data set and visualizing the
performance of the predicted masks.

There are two main functions:
    - calculate_and_show_metrics
    - show_overlap
    
First function calculates the mean Dice and identifies the top performing garments as well as
the lowest performing garments. Additionally highlights the difference between 
ground truth mask and predicted mask by:
    1. showing all pixel differences between the two masks
    2. showing pixel precent in ground truth but not in predicted mask
    3. showing pixel precent in predicted mask but not in ground truth
Finally a list of all Dice scores for supplied data set are displayed.

Second function applies an overlay in form of the predicted mask on top of the original image 
to visualize the performance of the predictor
"""

import cv2
import glob
from statistics import mean
import matplotlib.pyplot as plt  
import numpy as np
from tqdm import tqdm
import skimage
from sklearn.metrics import f1_score
    
import predict_single_pass as predict_single_pass

import util.params as params

input_size = params.input_size
orig_width = params.orig_width
orig_height = params.orig_height
batch_size = params.batch_size
threshold = params.threshold

model = params.model_factory()
# Will use the single pass predictor so NO focus path used
model.load_weights(filepath = params.weight_no_focus_path)


def load_images(img_path):    
    if not img_path.endswith("/*.*"): img_path = img_path + "/*.*"
    img_file_paths = sorted(glob.glob(img_path))
    img_list = []
    for file in tqdm(img_file_paths):
        img = cv2.imread(file)
        img_list.append(img)
    return img_list, img_file_paths 
 
"""
Generated the predicted masks using the single pass predictor
"""
def calc_predictions(img_orig_path):    
    orig_img_list, img_file_paths = load_images(img_orig_path)   
    prediction_list = []
    for img in tqdm(orig_img_list):
        mask = predict_single_pass.generate_mask(img)
        prediction_list.append(mask)
    
    prediction_list = np.squeeze(prediction_list, axis=3) # remove channel ex. (4, 1024, 1024, 1) --> (4, 1024, 1024)
    return prediction_list, img_file_paths

"""
Calculates the Dice (f1) score based on prediction and ground truth lists
"""
def calc_dice(prediction_list, mask_list):
    dice_map = {}
    for index in tqdm(range(len(prediction_list))):
        mask_predicted = prediction_list[index] > threshold 
        mask_ground_truth = cv2.cvtColor(mask_list[index], cv2.COLOR_BGR2GRAY)
        mask_ground_truth = mask_ground_truth >= 1        
        dice_score = f1_score(mask_ground_truth.flatten(), mask_predicted.flatten(), average='binary')
        dice_map[index] = dice_score
    return dice_map

"""
Extract filename from list index
"""
def file_name(img_file_paths, index):
    name = img_file_paths[index].replace("/", "\\")
    name = name.split('\\')[-1]
    return name

"""
Calculates the mean Dice and identifies the top performing garments as well as
the lowest performing garments. Additionally highlights the difference between 
ground truth mask and predicted mask by:
    1. showing all pixel differences between the two masks
    2. showing pixel precent in ground truth but not in predicted mask
    3. showing pixel precent in predicted mask but not in ground truth
Finally a list of all Dice scores for supplied data set are displayed.
"""
def calculate_and_show_metrics(img_orig_path, img_mask_path):    
    prediction_list, file_paths = calc_predictions(img_orig_path)
    mask_list, mask_paths  = load_images(img_mask_path)

    dice_map = calc_dice(prediction_list, mask_list)

    print("----------------------------------")
    print("-------------Metrics--------------")
    print("Data image set size: " + str(len(file_paths)))
    print("----------------------------------")

    print("Mean Dice coefficient: " + str(mean(dice_map.values())))    
    print("----------------------------------")


    print("Top 3 Dice: ")  
    top_dices = sorted(dice_map.keys(), key=dice_map.get, reverse=True)[0 : 3]
    for index in top_dices: 
        print("File: " + file_name(file_paths, index) + " has score: " + str(dice_map[index]))
    print("----------------------------------")

    print("Bottom 3 Dices: ")  
    bottom_dices = sorted(dice_map.keys(), key=dice_map.get)[0 : 3]
    for index in bottom_dices: 
        print("File: " + file_name(file_paths, index) + " has score: " + str(dice_map[index]))
    print("----------------------------------")
     
    print("Show Top Dice differences: ")  
    for index in top_dices: 
        fig = plt.figure(figsize=(18, 18))
        total_diff, groundtruth_diff, predicted_diff = show_difference_in_masks(index, prediction_list, mask_list)

        imgplt = fig.add_subplot(1, 3, 1)
        imgplt.set_title("Garment: " + file_name(file_paths, index))
        plt.imshow(total_diff)

        imgplt = fig.add_subplot(1, 3, 2)
        imgplt.set_title("Ground truth minus Predicted")
        plt.imshow(groundtruth_diff)

        imgplt = fig.add_subplot(1, 3, 3)
        imgplt.set_title("Predicted minus Ground truth")
        plt.imshow(predicted_diff)

        plt.show()
    print("----------------------------------")

    print("Show bottom Dice differences: ")  
    for index in bottom_dices: 
        fig = plt.figure(figsize=(18, 18))
        total_diff, groundtruth_diff, predicted_diff = show_difference_in_masks(index, prediction_list, mask_list)

        imgplt = fig.add_subplot(1, 3, 1)
        imgplt.set_title("Garment: " + file_name(file_paths, index))
        plt.imshow(total_diff)

        imgplt = fig.add_subplot(1, 3, 2)
        imgplt.set_title("Ground truth minus Predicted")
        plt.imshow(groundtruth_diff)

        imgplt = fig.add_subplot(1, 3, 3)
        imgplt.set_title("Predicted minus Ground truth")
        plt.imshow(predicted_diff)

        plt.show()
    print("----------------------------------")
    
    print("All Dice scores: ")  
    all_dices = sorted(dice_map.keys(), key=dice_map.get, reverse=True)
    for index in all_dices: 
        print(file_name(file_paths, index) + ", " + str(dice_map[index]))
    print("----------------------------------")
    
    
"""
Highlights the difference between ground truth mask and predicted mask by:
    1. showing all pixel differences between the two masks
    2. showing pixel precent in ground truth but not in predicted mask
    3. showing pixel precent in predicted mask but not in ground truth
"""
def show_difference_in_masks(index, prediction_list, mask_list):
    prediction = cv2.resize(prediction_list[index], (orig_width, orig_height))
    mask_predicted = prediction > threshold 

    mask_ground_truth = cv2.cvtColor(mask_list[index], cv2.COLOR_BGR2GRAY)
    mask_ground_truth = mask_ground_truth > 127
    
    mask_ground_truth = skimage.img_as_ubyte(mask_ground_truth, force_copy=False)
    mask_predicted = skimage.img_as_ubyte(mask_predicted, force_copy=False)
    total_diff_img = mask_ground_truth != mask_predicted
    groundtruth_diff_img = cv2.subtract(mask_ground_truth, mask_predicted)
    predicted_diff_img = cv2.subtract(mask_predicted, mask_ground_truth)
    return total_diff_img, groundtruth_diff_img, predicted_diff_img


"""
Overlays the prdicted mask on the original image to visualize the performance
of the predictor
"""
def show_overlap(img_orig_path):    
    img_list, img_paths  = load_images(img_orig_path)
    prediction_list, file_paths = calc_predictions(img_orig_path)
    
    for index in range(len(img_list)): 

        fig = plt.figure(figsize=(20, 20))
        imgplt = fig.add_subplot(1, 1, 1)
        imgplt.set_title("Original mask" + file_name(img_paths, index))
        predict_mask = prediction_list[index]
        predict_mask = predict_mask > threshold
        predict_mask= skimage.img_as_ubyte(predict_mask, force_copy=False)
        predict_mask = np.expand_dims(predict_mask, axis=2)
        predict_mask = cv2.cvtColor(predict_mask, cv2.COLOR_GRAY2RGB)      
        predict_mask[np.where((predict_mask > [0,0,0]).all(axis = 2))] = [0,255,0]
        
        original_img = img_list[index]
        
        img_with_predict_overlay = cv2.addWeighted(original_img, 0.7 , predict_mask, 0.3, -1)
        plt.imshow(img_with_predict_overlay)

        plt.show()
  


""" 
Demo calculating Dice for image set and visualizing results
"""
if __name__ == '__main__':
    
    img_orig_path = "./input/demonstration_set/original/*.*"
    mask_path = "./input/demonstration_set/mask/*.*"
        
    calculate_and_show_metrics(img_orig_path, mask_path)
    show_overlap(img_orig_path)