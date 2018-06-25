import cv2
import glob
import numpy as np
import skimage
from sklearn.metrics import f1_score
    
from util.remove_background import focus_garment_image
import predict_single_pass as predict_single_pass
import util.params as params

input_size = params.input_size
orig_width = params.orig_width
orig_height = params.orig_height
batch_size = params.batch_size
threshold = params.threshold
threshold_when_focused = params.threshold_when_focused

model = params.model_factory()
model.load_weights(filepath = params.weight_with_focus_path)


def generate_initial_mask(img):    
    return predict_single_pass.generate_mask(img)


def generate_final_mask(img, mask):  
    mask = np.expand_dims(mask, axis=2)
    mask = skimage.img_as_ubyte(mask, force_copy=False)
    
    focused_img = focus_garment_image(img, mask)
    focused_img = cv2.resize(focused_img, (input_size, input_size),  interpolation=cv2.INTER_AREA)
    focused_img = np.array(focused_img, np.float32) / 255
    focused_img = np.expand_dims(focused_img, axis=0)

    prediction = model.predict(focused_img)
    prediction = np.squeeze(prediction, axis=(0))
    prediction = cv2.resize(prediction, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
    prediction = np.expand_dims(prediction, axis=2)
    prediction = prediction > threshold_when_focused
    return prediction


"""
Our mask generator for double pass prediction, that is, working on imagery 
with has a reduce background.
"""
def generate_mask(img):
    initial_prediction = generate_initial_mask(img)
    initial_mask = initial_prediction > threshold
    prediction = generate_final_mask(img, initial_mask)
    return prediction


"""
Utility for showing performance of predictor in terms of Dice score
"""
def show_metrics_for_mask(img_name, img, mask):
    mask = mask >= 1
    initial_prediction = generate_initial_mask(img)
    initial_mask = initial_prediction > threshold
    single_dice = str(f1_score(mask.flatten(), initial_mask.flatten(), average='binary'))

    final_mask = generate_final_mask(img, initial_mask)
    final_dice = str(f1_score(mask.flatten(), final_mask.flatten(), average='binary'))

    print(img_name + ', ' + single_dice + ', ' + final_dice)


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