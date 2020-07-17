# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:22:14 2020

@author: Ching-Ting Kurt Lin

Evaluation matrices of the Unet3D model, including loss function, dice coefficient and IoU(Intersection of Union).
"""

def loss_func(y_true, y_pred):
    from keras import backend as K
    
    def dice_loss(y_true, y_pred, smooth=1):
        intersection = K.sum(y_true * y_pred, axis=[1,2,3,4])
        union = K.sum(y_true, axis=[1,2,3,4]) + K.sum(y_pred, axis=[1,2,3,4])
        dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
        return -K.log(dice)  
    
    return K.categorical_crossentropy(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1):
    from keras import backend as K
    
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.
        """
        true_positives = K.sum(K.round(K.clip(y_true[:,:,:,0] * y_pred[:,:,:,0], 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[:,:,:,0], 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.
        """
        true_positives = K.sum(K.round(K.clip(y_true[:,:,:,0] * y_pred[:,:,:,0], 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred[:,:,:,0], 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ( (precision * recall) / (precision + recall + K.epsilon()) )
    
def IoU(y_true, y_pred, smooth=1):
    from keras import backend as K
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3,4])
    union = K.sum(y_true,[1,2,3,4])+K.sum(y_pred,[1,2,3,4])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou
