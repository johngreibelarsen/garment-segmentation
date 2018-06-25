import cv2
import glob
import numpy as np
from sklearn.metrics import f1_score

import util.params as params


input_size = params.input_size
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold

centering_mean_path = params.centering_mean_path
normalizing_std_path = params.normalizing_std_path

model = params.model_factory()
model.load_weights(filepath = params.weight_no_focus_path)

"""
Our mask generator for single pass prediction, that is, directly on the original
image (no focus to reduce background)
"""
def generate_mask(img):    
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_AREA)
       
    img = np.array(img, np.float32) / 255
    img = np.expand_dims(img, axis=0)

    img_mean = np.load(centering_mean_path)
    img_std = np.load(normalizing_std_path)

    img -= img_mean # zero-center
    img /= img_std # normalize

    prediction = model.predict(img)
    prediction = np.squeeze(prediction, axis=(0, 3))
    prediction = cv2.resize(prediction, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
    return prediction

 
"""
Utility for showing performance of predictor in terms of Dice score
"""
def show_metrics_for_mask(img_name, img, mask):
    predicted_mask = generate_mask(img)
    predicted_mask = predicted_mask > threshold
    mask = mask >= 1
    dice_score = str(f1_score(mask.flatten(), predicted_mask.flatten(), average='binary'))
    print(img_name + ', ' + dice_score)


"""
Demo showing performance of predictor in terms of Dice score
"""
if __name__ == '__main__':
    img_orig_path = "./input/demonstration_set/original/*.*"
    img_mask_path = "./input/demonstration_set/mask/*.*"

    img_path_list = sorted(glob.glob(img_orig_path))
    mask_path_list = sorted(glob.glob(img_mask_path))
    
    for index in range(0, len(img_path_list)):
        img_orig = cv2.imread(img_path_list[index])
        img_mask = cv2.imread(mask_path_list[index], 0) # Reas as gray
        show_metrics_for_mask(img_path_list[index], img_orig, img_mask)