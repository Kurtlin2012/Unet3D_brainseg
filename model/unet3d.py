#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:22:37 2020

@author: Ching-Ting Kurt Lin

This script is to create a Unet3D model.
"""

def unet3d(input_shape=[256,256,64,1], channels=10, pretrained_weights=None, learning_rate=1e-3):
    
    """
    Input:
        input_shape: numpy
            The shape of input matrix [height, width, depth, channel=1]. Default is [256,256,64,1].
        channels: int
            The number of channels of the output. Default is 10.
        pretrained_weights: .h5 file
            Load the pretrained weights if it exists. Default is None.
        learning_rate: float
            Change the initial learning rate of the model. Default is 1e-3.

    Output:
        model:
            The Unet3D model.
    """
    
    import keras
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Concatenate, BatchNormalization, Activation, Conv3D, UpSampling3D, MaxPooling3D
    from keras.optimizers import Adam
    from ..model.evalmatrix import loss_func, dice_coef, IoU
    
    inputs = Input(shape=input_shape)
    enc = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(inputs)
    temp = enc
    enc = BatchNormalization()(enc)
    enc = Activation("selu")(enc)
    enc = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(enc)
    enc = BatchNormalization()(enc)
    enc = keras.layers.Add()([enc, temp])
    enc = Activation("selu")(enc)
    del temp
    
    enc2 = MaxPooling3D(pool_size=(2,2,2))(enc)
    enc2 = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(enc2)
    temp = enc2
    enc2 = BatchNormalization()(enc2)
    enc2 = Activation("selu")(enc2)
    enc2 = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(enc2)
    enc2 = BatchNormalization()(enc2)
    enc2 = keras.layers.Add()([enc2, temp])
    enc2 = Activation("selu")(enc2)
    del temp
    
    enc3 = MaxPooling3D(pool_size=(2,2,2))(enc2)
    enc3 = Conv3D(filters=128, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(enc3)
    temp = enc3
    enc3 = BatchNormalization()(enc3)
    enc3 = Activation("selu")(enc3)
    enc3 = Conv3D(filters=128, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(enc3)
    enc3 = BatchNormalization()(enc3)
    enc3 = keras.layers.Add()([enc3, temp])
    enc3 = Activation("selu")(enc3)
    del temp
    
    enc4 = MaxPooling3D(pool_size=(2,2,2))(enc3)
    enc4 = Conv3D(filters=256, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(enc4)
    temp = enc4
    enc4 = BatchNormalization()(enc4)
    enc4 = Activation("selu")(enc4)
    enc4 = Conv3D(filters=256, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(enc4)
    enc4 = BatchNormalization()(enc4)
    enc4 = keras.layers.Add()([enc4, temp])
    enc4 = Activation("selu")(enc4)
    del temp
    
    dec = UpSampling3D(size=(2,2,2))(enc4)
    dec = Concatenate(axis=-1)([dec, enc3])
    dec = Conv3D(filters=256, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(dec)
    temp = dec
    dec = BatchNormalization()(dec)
    dec = Activation("selu")(dec)
    dec = Conv3D(filters=256, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(dec)
    dec = BatchNormalization()(dec)
    dec = keras.layers.Add()([dec, temp])
    dec = Activation("selu")(dec)
    del temp
    
    dec2 = UpSampling3D(size=(2,2,2))(dec)
    dec2 = Concatenate(axis=-1)([dec2, enc2])
    dec2 = Conv3D(filters=128, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(dec2)
    temp = dec2
    dec2 = BatchNormalization()(dec2)
    dec2 = Activation("selu")(dec2)
    dec2 = Conv3D(filters=128, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(dec2)
    dec2 = BatchNormalization()(dec2)
    dec2 = keras.layers.Add()([dec2, temp])
    dec2 = Activation("selu")(dec2)
    del temp
    
    dec3 = UpSampling3D(size=(2,2,2))(dec2)
    dec3 = Concatenate(axis=-1)([dec3, enc])
    dec3 = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(dec3)
    temp = dec3
    dec3 = BatchNormalization()(dec3)
    dec3 = Activation("selu")(dec3)
    dec3 = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='glorot_normal')(dec3)
    dec3 = BatchNormalization()(dec3)
    dec3 = keras.layers.Add()([dec3, temp])
    dec3 = Activation("selu")(dec3)
    del temp
    
    out = Conv3D(filters=channels, kernel_size=(3,3,3), activation='softmax', padding='same', kernel_initializer='glorot_normal', name="seg_output")(dec3)
    
    model = Model(inputs,out)
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss_func, metrics=[dice_coef, IoU])
    if pretrained_weights:
        model.load_weight(pretrained_weights)

    return model
