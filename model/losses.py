""" 
    Implementation of our Dice coefficient and associated loss function 
"""

from keras.losses import binary_crossentropy
import keras.backend as K


def dice_coeff(y_true, y_pred):
    """
        We measure our accuracy against the Dice Coefficient (F1):
        	2∗|X∩Y|/(|X|+|Y|)       or       2TP/(2TP+FP+FN)
        X being our prediction matrix and Y our target matrix – basically the 
        intersection over the union.
    """
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def bce_dice_loss(y_true, y_pred):
    """
        The loss function we use to traing the model on by keeping decreasing 
        this function.
        Note for we weight the dice coefficient over the BCE
    """
    loss = 0.5 * binary_crossentropy(y_true, y_pred) - 2*dice_coeff(y_true, y_pred)
    return loss
