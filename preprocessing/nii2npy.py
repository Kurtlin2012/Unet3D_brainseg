# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:30:30 2020

@author: Ching-Ting Kurt Lin
"""

def nii2npy(ori_folder, gt_folder, output_folder, resolution = [256,256,64]):
    
    """
    Input:
        ori_folder: string
            Folder path of the training data.
        gt_folder: string
            Folder path of the ground truth.
        output_folder: string
            Folder path to keep the combined matrices.
        resolution: list
            Set the shape of the 3D matrix. The input list should be [H(height), W(width), D(depth)].
    
    Output:
        X: 5-D numpy matrix
            Combined matrix of all the training data.
        Y: 5-D numpy matrix
            Combined matrix of all the grouth truth.
    
    """
    
    import os
    import numpy as np
    import nibabel as nib
    import glob
    from skimage.transform import resize
    
    os.chdir(ori_folder)
    folder_list = os.listdir()
    folder_list.sort(key=lambda x:x)
    X = np.zeros([len(folder_list), resolution[0], resolution[1], resolution[2]], dtype=float)
    
    # Original image processing
    for i in range(len(folder_list)):
        os.chdir(ori_folder + '/' + folder_list[i])
        file = glob.glob('*.nii')
        img_data = nib.load(file[0]).get_fdata()
        img_data = np.swapaxes(img_data,0,1)
        img_data = np.flip(img_data,0)
        img = resize(img_data, resolution)
        
        # Gaussian Normalization
        img = (img-img.mean()) / img.std()
        
        # Combining all files
        if i+1//10 == i+1/10:
            print('Ori: Folder no.' + str(i+1))
        X[i,:,:,:] = img
        del img
    
    X = np.expand_dims(X, axis=4)
    os.chdir(output_folder)
    np.save('X.npy', X)
    print('All 3D images were combined together, next...')
    del X
    
    # Ground truth processing
    os.chdir(gt_folder)
    file_size = len(os.listdir(gt_folder + '/' + folder_list[0]))
    Y = np.zeros([len(folder_list), resolution[0], resolution[1], resolution[2], file_size], dtype=bool)
    
    for i in range(len(folder_list)):
        os.chdir(gt_folder + '/' + folder_list[i])
        file_list = glob.glob('*.nii')
        file_list.sort(key=lambda x:x)
        img_temp = np.zeros([resolution[0], resolution[1], resolution[2], len(file_list)], dtype=bool)
        for j in range(len(file_list)):
            img_data = nib.load(file_list[j]).get_fdata()
            img_data = np.swapaxes(img_data,0,1)
            img_data = np.flip(img_data,0)
            img = resize(img_data, resolution)
            img_temp[:,:,:,j] = img[:,:,:,0]
            
        Y[i,:,:,:,:] = img_temp
        if i+1//10 == i+1/10:
            print('GT: Folder no.' + str(i+1))
        
    os.chdir(output_folder)
    np.save('Y.npy', Y)
    print('Ground truth of the 3D images were combine together.')
    del Y
