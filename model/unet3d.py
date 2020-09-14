#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:22:37 2020

@author: Ching-Ting Kurt Lin

This script is to create a Unet3D model.
"""

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Concatenate, BatchNormalization, Activation, Conv3D, UpSampling3D, MaxPooling3D
from keras.optimizers import Adam

class unet_model(object):
        
    def __init__(self, input_shape=[256,256,64,1], channels=10, pretrained_weights=None, learning_rate=1e-3, start_filter=32):
        self.input_shape = input_shape
        self.channels = channels
        self.pretrained_weights = pretrained_weights
        self.loss = self.loss_func
        self.metrics = [self.dice_coef, self.IoU, 'accuracy']
        self.optimizer = Adam(lr = learning_rate)
        self.init_filter = start_filter
        self.model = self.unet3d()


    def unet3d(self):
        inputs = Input(shape=self.input_shape)
        enc = Conv3D(filters=self.init_filter, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(inputs)
        temp = inputs
        enc = BatchNormalization()(enc)
        enc = Activation("selu")(enc)
        enc = Conv3D(filters=self.init_filter, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(enc)
        enc = BatchNormalization()(enc)
        enc = keras.layers.Add()([enc, temp])
        enc = Activation("selu")(enc)
        del temp
        
        enc2 = MaxPooling3D(pool_size=(2,2,2))(enc)
        enc2 = Conv3D(filters=self.init_filter * 2, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(enc2)
        temp = enc2
        enc2 = BatchNormalization()(enc2)
        enc2 = Activation("selu")(enc2)
        enc2 = Conv3D(filters=self.init_filter * 2, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(enc2)
        enc2 = BatchNormalization()(enc2)
        enc2 = keras.layers.Add()([enc2, temp])
        enc2 = Activation("selu")(enc2)
        del temp
        
        enc3 = MaxPooling3D(pool_size=(2,2,2))(enc2)
        enc3 = Conv3D(filters=self.init_filter * 4, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(enc3)
        temp = enc3
        enc3 = BatchNormalization()(enc3)
        enc3 = Activation("selu")(enc3)
        enc3 = Conv3D(filters=self.init_filter * 4, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(enc3)
        enc3 = BatchNormalization()(enc3)
        enc3 = keras.layers.Add()([enc3, temp])
        enc3 = Activation("selu")(enc3)
        del temp
        
        enc4 = MaxPooling3D(pool_size=(2,2,2))(enc3)
        enc4 = Conv3D(filters=self.init_filter * 8, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(enc4)
        temp = enc4
        enc4 = BatchNormalization()(enc4)
        enc4 = Activation("selu")(enc4)
        enc4 = Conv3D(filters=self.init_filter * 8, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(enc4)
        enc4 = BatchNormalization()(enc4)
        enc4 = keras.layers.Add()([enc4, temp])
        enc4 = Activation("selu")(enc4)
        del temp
        
        dec = UpSampling3D(size=(2,2,2))(enc4)
        dec = Concatenate(axis=-1)([dec, enc3])
        dec = Conv3D(filters=self.init_filter * 8, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(dec)
        temp = dec
        dec = BatchNormalization()(dec)
        dec = Activation("selu")(dec)
        dec = Conv3D(filters=self.init_filter * 8, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(dec)
        dec = BatchNormalization()(dec)
        dec = keras.layers.Add()([dec, temp])
        dec = Activation("selu")(dec)
        del temp
        
        dec2 = UpSampling3D(size=(2,2,2))(dec)
        dec2 = Concatenate(axis=-1)([dec2, enc2])
        dec2 = Conv3D(filters=self.init_filter * 4, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(dec2)
        temp = dec2
        dec2 = BatchNormalization()(dec2)
        dec2 = Activation("selu")(dec2)
        dec2 = Conv3D(filters=self.init_filter * 4, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(dec2)
        dec2 = BatchNormalization()(dec2)
        dec2 = keras.layers.Add()([dec2, temp])
        dec2 = Activation("selu")(dec2)
        del temp
        
        dec3 = UpSampling3D(size=(2,2,2))(dec2)
        dec3 = Concatenate(axis=-1)([dec3, enc])
        dec3 = Conv3D(filters=self.init_filter * 2, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(dec3)
        temp = dec3
        dec3 = BatchNormalization()(dec3)
        dec3 = Activation("selu")(dec3)
        dec3 = Conv3D(filters=self.init_filter * 2, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(dec3)
        dec3 = BatchNormalization()(dec3)
        dec3 = keras.layers.Add()([dec3, temp])
        dec3 = Activation("selu")(dec3)
        del temp
        
        out = Conv3D(filters=self.channels, kernel_size=(3,3,3), activation='softmax', padding='same', kernel_initializer='glorot_normal', name="seg_output")(dec3)
        
        model = Model(inputs,out)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        # model.summary()
        if self.pretrained_weights:
            model.load_weight(self.pretrained_weights)
        return model
    
    
    def loss_func(self, y_true, y_pred, smooth=0.01):
        
        def dice_loss(self, y_true, y_pred, smooth=1, axis=[1,2,3,4]):
            intersection = K.sum(y_true * y_pred, axis=axis)
            union = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis)
            dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
            return -K.log(dice)  
        
        return dice_loss(self, y_true, y_pred) + K.categorical_crossentropy(y_true, y_pred)
    
    
    def dice_coef(self, y_true, y_pred):
        
        def recall(self, y_true, y_pred):
            """Recall metric.
    
            Only computes a batch-wise average of recall.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall
    
        def precision(self, y_true, y_pred):
            """Precision metric.
    
            Only computes a batch-wise average of precision.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        
        precision = precision(self, y_true, y_pred)
        recall = recall(self, y_true, y_pred)
        return 2 * ( (precision * recall) / (precision + recall + K.epsilon()) )
    
    
    def IoU(self, y_true, y_pred, axis=[1,2,3,4], smooth=0.01):
        intersection = K.sum(K.abs(y_true * y_pred), axis=axis)
        union = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) - intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou
