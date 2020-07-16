#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Juy 1 14:55:55 2020

@author: Ching-Ting Kurt Lin
"""

import os
import glob
import numpy as np
from keras.models import load_model
import nibabel as nib
from .model.showfigures import plot_transform, unet3d_report
from .preprocessing.tools import categorise_prediction
from .model.evalmatrix import loss_func, dice_coef, IoU

#%% 
def unet3d_predict(weight_dir, X_dir, image_folder, output_folder, channel_order):
    """
    Args:
    weight_dir: string
        Path of the weight(.h5 file).
    X: numpy
        Path of the numpy file. The shape of the matrix will be [number, height, width, depth, channel=1].
    image_folder: string
        Path of the folder of original images. It will be used to check the original nifti file's header. Need to confirm if the order of the folder is same to the 3D numpy matrix(X).
    output_folder: string
        Path of the folder to store the output plots.
    channel_order: list, optional
        The channel of LCRB, LGM, LWM, RCRB, RGM, RWM. The default is [3,4,1,7,8,5].
    """
    # input the order of LCRB, LGM, LWM, RCRB, RGM, RWM channels
    ask = input('Do you want to change the order of the channel?(Y/N)')
    if ask == 'Y' or ask == 'y':
        channel_order[0] = int(input('Input the channel of LCRB: '))
        channel_order[1] = int(input('Input the channel of LGM: '))
        channel_order[2] = int(input('Input the channel of LWM: '))
        channel_order[3] = int(input('Input the channel of RCRB: '))
        channel_order[4] = int(input('Input the channel of RGM: '))
        channel_order[5] = int(input('Input the channel of RWM: '))
    else:
        channel_order=[3,4,1,7,8,5]
    
    # load the model
    model = load_model(weight_dir, custom_objects={'loss_func': loss_func, 'dice_coef': dice_coef, 'IoU': IoU}, compile=False)
    
    # load the 3d image matrix
    X = np.load(X_dir)

    folder_list = os.listdir(image_folder)
    for i in range(len(folder_list)):
        # get the pixel spacing of 3 dimensions
        os.chdir(image_folder + '/' + folder_list[i])
        X_filename = glob.glob('*.nii')
        X_header = nib.load(X_filename[0]).header
        voxel = (X_header['dim'][1] * X_header['pixdim'][1]) * (X_header['dim'][2] * X_header['pixdim'][2]) * (X_header['dim'][3] * X_header['pixdim'][3]) / (X.shape[1] * X.shape[2] * X.shape[3])
        
        # predict one case each time
        X_test = np.expand_dims(X[i,:,:,:,:], axis=0)
        y_temp = model.predict(X_test)
        y_pred = np.zeros([1, y_temp.shape[1], y_temp.shape[2], y_temp.shape[3], y_temp.shape[4]], dtype=float)
        for j in range(y_temp.shape[-1]):
            y_temp2 = categorise_prediction(np.argmax(y_temp, axis = 4), j)
            y_temp2 = np.expand_dims(y_temp2, axis = 4)
            y_pred[0,:,:,:,j] = y_temp2
            del y_temp2
        del y_temp
        name_ord = X_filename.find('.nii')
        output_image = output_folder + '/' + X_filename[:name_ord] + '.png'
        
        # plot the image
        unet3d_report(X_test, y_pred, output_image, voxel, channel_order)
