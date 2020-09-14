# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:30:30 2020

@author: Ching-Ting Kurt Lin
"""

"""

Output:
    X: 5-D numpy matrix
        Combined matrix of all the training data.
    Y: 5-D numpy matrix
        Combined matrix of all the grouth truth.
        
"""

import os
import argparse
import sys
import numpy as np
import nibabel as nib
import glob
from skimage.transform import resize
from tools import categorise_prediction

def parse_args(args):
    parser = argparse.ArgumentParser(description='Data transformation (Nifti to numpy)')
    parser.add_argument('--image', type = str, help = 'Folder path of the original data.')
    parser.add_argument('--label', type = str, help = 'Folder path of the label/ground truth.')
    parser.add_argument('--out', type = str, help = 'Folder path to keep the combined matrices.')
    parser.add_argument('--reso', type = list, default = [256, 256, 64], help = 'Set the shape of the 3D matrix. The input list should be [H(height), W(width), D(depth)].')
    return parser.parse_args(args)


def main(args = None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    folder_list = os.listdir(args.image)
    folder_list.sort(key=lambda x:x)
    X = np.zeros([len(folder_list), args.reso[0], args.reso[1], args.reso[2]], dtype=float)
    
    # Original image processing
    for i in range(len(folder_list)):
        os.chdir(args.image + '/' + folder_list[i])
        file = glob.glob('*.nii')
        img_data = nib.load(file[0]).get_fdata()
        img_data = np.swapaxes(img_data,0,1)
        img_data = np.flip(img_data,0)
        img = resize(img_data, args.reso)
        
        # Gaussian Normalization
        img = (img-img.mean()) / img.std()
        
        # Combining all files
        if i+1//10 == i+1/10:
            print('Ori: Folder no.' + str(i+1))
        X[i,:,:,:] = img
        del img
    
    X = np.expand_dims(X, axis=4)
    np.save(args.out + '/X.npy', X)
    print('All 3D images were combined together, next...')
    del X
    
    # Ground truth processing
    file_size = len(os.listdir(args.label + '/' + folder_list[0]))
    
    Y = np.zeros([len(folder_list), args.reso[0], args.reso[1], args.reso[2], file_size], dtype=bool)
    
    for i in range(len(folder_list)):
        os.chdir(args.label + '/' + folder_list[i])
        file_list = glob.glob('*.nii')
        file_list.sort(key=lambda x:x)
        img_temp = np.zeros([args.reso[0], args.reso[1], args.reso[2], len(file_list)], dtype=bool)
        for j in range(len(file_list)):
            img_data = nib.load(file_list[j]).get_fdata()
            img_data = np.swapaxes(img_data,0,1)
            img_data = np.flip(img_data,0)
            img = resize(img_data, args.reso)
            img_temp[:,:,:,j] = img[:,:,:,0]
        img_temp = np.expand_dims(img_temp, axis=0)
            
        # Categorize prediction to prevent if the voxel was classified into multiple channels.
        y = np.zeros(img_temp.shape, dtype=float)
        for j in range(y.shape[-1]):
            y_temp = categorise_prediction(np.argmax(y, axis = 4), j)
            y_temp = np.expand_dims(y_temp, axis = 4)
            y[0,:,:,:,j] = y_temp
            del y_temp
        
        Y[i,:,:,:,:] = y[0,:,:,:,:]
        
        if i+1//10 == i+1/10:
            print('GT: Folder no.' + str(i+1))
            
    np.save(args.out + '/Y.npy', Y)
    print('Ground truth of the 3D images were combine together.')
    del Y
    
if __name__ == '__main__':
    main()