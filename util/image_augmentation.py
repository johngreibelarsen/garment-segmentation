"""
In our code, we’re using the functions from petrosgk’s Carvana example 
(https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/blob/master/train.py) 
to randomly alter the hue, saturation, and value of the image (HSV color space), 
and to randomly shift, scale, rotate as well as horisontally flip the image.
"""
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt  

import params


input_size = params.input_size
batch_size = params.batch_size


"""
Randomly alter the hue (color), saturation (grayness), and value (brightness) of the 
image (HSV color space)
"""
def randomHueSaturationValue(image, hue_shift_limit=(0, 179), sat_shift_limit=(0, 255),
                             val_shift_limit=(0, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


"""
Randomly shift, scale (zoom) and rotate imagery
"""
def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0,0,0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(0, 0, 0,))

    return image, mask


"""
Randomly flip image horizontally
"""
def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


"""
Training generator using all of the above image manipulations to generate 
augmented images
"""
def train_generator(img_fnames, seg_fnames):
    while True:
        for start in range(0, len(img_fnames), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(img_fnames))
            for index in range(start, end):
                img = cv2.imread(img_fnames[index])
                img = cv2.resize(img, (input_size, input_size))                
                seg = cv2.imread(seg_fnames[index], cv2.IMREAD_GRAYSCALE)
                seg = cv2.resize(seg, (input_size, input_size))
                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
                img, mask = randomShiftScaleRotate(img, seg,
                                                   shift_limit=(-0.01, 0.01),
                                                   scale_limit=(-0.1, 0.1),
                                                   rotate_limit=(-5, 5))
                img, seg = randomHorizontalFlip(img, seg)
                seg = np.expand_dims(seg, axis=2)
                x_batch.append(img)
                y_batch.append(seg)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


"""
This is the augmentation configuration we will use for validation. Note there are no 
data augmentation for validation data             
"""
def valid_generator(img_fnames, seg_fnames):
    while True:
        for start in range(0, len(img_fnames), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(img_fnames))
            for index in range(start, end):
                img = cv2.imread(img_fnames[index])
                img = cv2.resize(img, (input_size, input_size))
                seg = cv2.imread(seg_fnames[index], cv2.IMREAD_GRAYSCALE)
                seg = cv2.resize(seg, (input_size, input_size))
                seg = np.expand_dims(seg, axis=2)
                x_batch.append(img)
                y_batch.append(seg)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch      
            
        
"""
Test training generator for visualizing the effect of augmentation on our imagery
"""
def visualize_train_generator(img_fnames, seg_fnames):
    orig_list = []
    orig_mask_list = []    
    argmented_list = []
    argmented_mask_list = []
    
    for index in range(0, len(img_fnames)):
        img = cv2.imread(img_fnames[index])
        img = cv2.resize(img, (input_size, input_size))                
        orig_list.append(img.copy())
        
        seg = cv2.imread(seg_fnames[index], cv2.IMREAD_GRAYSCALE)
        seg = cv2.resize(seg, (input_size, input_size))
        orig_mask_list.append(seg.copy())
        img = randomHueSaturationValue(img,
                                       hue_shift_limit=(0, 40),
                                       sat_shift_limit=(0, 25),
                                       val_shift_limit=(0, 25))
        img, seg = randomShiftScaleRotate(img, seg,
                                           shift_limit=(-0.1, 0.1),
                                           scale_limit=(-0.1, 0.1),
                                           rotate_limit=(-5, 5))
        img, seg = randomHorizontalFlip(img, seg)
        #seg = np.expand_dims(seg, axis=2)
        argmented_list.append(img)
        argmented_mask_list.append(seg)
    return orig_list, orig_mask_list, argmented_list, argmented_mask_list


""" Visualize the effect of the test training generator """
if __name__ == '__main__':
    img_orig_path = "../input/demonstration_set/original/*.*"
    img_orig_list = sorted(glob.glob(img_orig_path))
    img_mask_path = "../input/demonstration_set/mask/*.*"
    img_mask_list = sorted(glob.glob(img_mask_path))
    for epocs in range(0, 3):        
        orig_list, orig_mask_list, argmented_list, argmented_mask_list = visualize_train_generator(img_orig_list, img_mask_list)
        for i in range(len(orig_list)):
            
            fig = plt.figure(figsize=(15, 15))
            imgplt = fig.add_subplot(1, 4, 1)
            imgplt.set_title("Original")           
            plt.imshow(orig_list[i])

            imgplt = fig.add_subplot(1, 4, 2)
            imgplt.set_title("Original mask")           
            plt.imshow(orig_mask_list[i])
                        
            imgplt = fig.add_subplot(1, 4, 3)
            imgplt.set_title("Augmented")
            plt.imshow(argmented_list[i])

            imgplt = fig.add_subplot(1, 4, 4)
            imgplt.set_title("Augmented mask")
            plt.imshow(argmented_mask_list[i])
            