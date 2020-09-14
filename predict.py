#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Juy 1 14:55:55 2020

@author: Ching-Ting Kurt Lin
"""

import os
import argparse
import sys
import numpy as np
from keras.models import load_model
import nibabel as nib
from .model.showfigures import unet3d_report
from tools import categorise_prediction

"""

Output:
    report: .png file
        A report will be generate that have preview of the result of segmentation including axial, coronal and sagittal planes, ICV(intracranial volume) and percentages of each part of the brain.

"""

def parse_args(args):
    parser = argparse.ArgumentParser(description='Unet3D prediction')
    parser.add_argument('--weight', type = str, help = 'File path of the pretrained weights(h5 file).')
    parser.add_argument('--test', type = str, help = 'Folder path of the testing data(nifti file).')
    parser.add_argument('--out', type = str, help = 'Folder path to save the generated reports.')
    return parser.parse_args(args)


def main(args = None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    # input the order of LCRB, LGM, LWM, RCRB, RGM, RWM channels
    channel_order = np.zeros([6,], dtype=int)
    ask = input('Do you want to change the order of the channel? (Y/N)')
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
    model = load_model(args.weight, compile=False)
    
    # load the 3d image matrix
    file_list = os.listdir(args.test)
    for i in range(len(file_list)):
        X = nib.load(file_list[i])
        X_header = X.header
        X = np.expand_dims(np.expand_dims(X, axis=0), axis=4)
        X = np.swapaxes(X, 0, 1)
        X = np.flip(X, 0)
        voxel = (X_header['dim'][1] * X_header['pixdim'][1]) * (X_header['dim'][2] * X_header['pixdim'][2]) * (X_header['dim'][3] * X_header['pixdim'][3]) / (X.shape[1] * X.shape[2] * X.shape[3])
        
        # predict one case each time
        y_pred = model.predict(X)
        y = np.zeros([1, y_pred.shape[1], y_pred.shape[2], y_pred.shape[3], y_pred.shape[4]], dtype=float)
        for j in range(y_pred.shape[-1]):
            y_temp = categorise_prediction(np.argmax(y_pred, axis = 4), j)
            y_temp = np.expand_dims(y_temp, axis = 4)
            y[0,:,:,:,j] = y_temp
            del y_temp
        del y_pred
        
        name_ord = file_list[i].find('.nii')
        output = args.out + '/' + file_list[i][:name_ord] + '.png'
        output_matrix = args.out + '/' + file_list[i][:name_ord] + '.npy'
        
        # Plot the image and save the predict result
        unet3d_report(X, y, output, voxel, channel_order)
        np.save(output_matrix, y)


if __name__ == '__main__':
    main()
