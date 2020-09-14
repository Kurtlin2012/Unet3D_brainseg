"""
Created on Fri Jun 17 09:48:50 2020

@author: Ching-Ting Kurt Lin
"""

############################ 3D Image Data Generator ############################
# This script can be used when the training data is not enough.                 
# It will read the numpy file for both image and ground truth made from nii2npy,
# then randomly shift, zoom in/out and rotate the 3d matrix.
# All the parameters will be saved when the program ends. 
#################################################################################

import numpy as np
import random
import scipy.ndimage as ndi
import csv
import argparse
import sys
from tools import categorise_prediction

"""
    
Output:
    new_img: 5-D Numpy
        Augmented MRI data. The shape of matrix is [number=1, hetght, width, depth, channel=1].            
    new_gt: 5-D Numpy
        Augmented ground truth. The shape of matrix is [number=1, hetght, width, depth, channel].
    para: List
        List of all parameters for each augmented data, including the index of original image, shift factor, zoom factor and rotate factor.

"""

def parse_args(args):
    parser = argparse.ArgumentParser(description='Data augmentation')
    parser.add_argument('--image', type = str, help = 'Folder path of the original data (5-D numpy matrix).')
    parser.add_argument('--label', type = str, help = 'Folder path of the label/ground truth (5-D numpy matrix).')
    parser.add_argument('--out', type = str, help = 'Folder path to keep the augment datas.')
    parser.add_argument('--num', type = int, default = 500, help = 'The amount of augmented datas.')
    parser.add_argument('--combine', type = str, default = True, help = 'Combine or separate the augment files (True/False). Need to check the limitation of the RAM while combining all files.')
    parser.add_argument('--flip', type = str, default = False, help = 'Enable/Disable the flip function (True/False).')
    parser.add_argument('--shiftran', type = int, default = 5, help = 'Setting the range of shifting pixels (only for x and y axis).')
    parser.add_argument('--zoomran', type = float, default = 1.1, help = 'Setting the range of zooming factor (default = 1).')
    parser.add_argument('--rotran', type = int, default = 5, help = 'Setting the range of rotating angle (degrees).')
    return parser.parse_args(args)


def main(args = None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    # load the numpy files
    X_ori = np.load(args.image)
    Y_ori = np.load(args.label)
    
    print('Data combining: ' + str(args.combine))
    
    # list for parameters
    param = [['No.', 'Original File Index', 'Flip axis', 'Shift Factor X', 'Shift Factor Y', 'Zoom Factor', 'Rotate Angle']]
    
    for i in range(args.num):
        # parameters
        img_order=random.randint(0,X_ori.shape[0]-1)
        shift_range_x=random.randint(-args.shiftran, args.shiftran)
        shift_range_y=random.randint(-args.shiftran, args.shiftran)
        zoom_range=random.uniform(1-abs(1-args.zoomran),1+abs(1-args.zoomran))
        rotate_angle=random.uniform(-args.rotran, args.rotran)
        
        # read original image
        ori_img = X_ori[img_order,:,:,:,0]
        new_img = ori_img
        xaxis = ori_img.shape[0]
        yaxis = ori_img.shape[1]
        zaxis = ori_img.shape[2]
        
        # combine or separate
        if args.combine == 'True':
            X_com = np.zeros([1, xaxis, yaxis, zaxis, 1])
        
        # flip
        if args.flip == 'True':
            flip_order = random.randint(0,2)
            new_img = np.flip(new_img, flip_order)
        
        # shift
        file_shift = np.zeros([xaxis+2*abs(shift_range_x),yaxis+2*abs(shift_range_y),zaxis],dtype=float)
        # file_shift += ori_img[0,0,0]
        for j in range(xaxis):
            for k in range(yaxis):
                file_shift[j+abs(shift_range_x)+shift_range_x,k+abs(shift_range_y)+shift_range_y,:] = new_img[j,k,:]
        new_img = file_shift[abs(shift_range_x):xaxis+abs(shift_range_x),abs(shift_range_y):yaxis+abs(shift_range_y),:]        
        
        # zoom
        file_zoom = ndi.zoom(new_img, zoom_range, order=0, mode='nearest')
        xs = file_zoom.shape[0]
        ys = file_zoom.shape[1]
        zs = file_zoom.shape[2]
        if zoom_range >= 1:
            xh = (xs-xaxis)//2
            yh = (ys-yaxis)//2
            zh = (zs-zaxis)//2
            new_img = file_zoom[xh:xh+xaxis, yh:yh+yaxis, zh:zh+zaxis]
        elif zoom_range < 1:
            xh = (xaxis-xs)//2
            yh = (yaxis-ys)//2
            zh = (zaxis-zs)//2
            file_fix = np.zeros(shape=[xaxis,yaxis,zaxis])
            # file_fix += ori_img[0,0,0]
            file_fix[xh:xh+xs,yh:yh+ys,zh:zh+zs]=file_zoom
            new_img = file_fix
            del file_fix
        
        # rotate
        new_img = ndi.rotate(new_img, rotate_angle, axes=(0,1), order=0, mode='nearest', reshape=False)
    
        # normalize and save
        new_img = (new_img-new_img.mean())/new_img.std()
        new_img = np.expand_dims(new_img, axis = 0)
        new_img = np.expand_dims(new_img, axis = 4)
        if args.flip == True:
            param.append([i+1, img_order+1, flip_order+1, shift_range_x, shift_range_y, zoom_range, rotate_angle])
        else:
            param.append([i+1, img_order+1, 0, shift_range_x, shift_range_y, zoom_range, rotate_angle])
        print('Generating augment X no.' + str(i+1))
        
        # save the augment image separately or combine all files together
        if args.combine == 'True':
            X_com = np.concatenate((X_com, new_img), axis=0)
        else:
            if len(str(i+1)) < len(str(args.num)):
                name = '0' * (len(str(args.num)) - len(str(i+1))) + str(i+1) + '.npy'
            else:
                name = str(i+1) + '.npy'
            np.save(args.out + '/X_' + name, new_img) 
        del new_img
        
        # read original ground truth
        ori_gt = Y_ori[img_order,:,:,:,:]
        new_gt = np.zeros(ori_gt.shape)
        if args.combine == 'True':
            Y_com = np.zeros([1, xaxis, yaxis, zaxis, ori_gt.shape[-1]], dtype='bool')

        for j in range(ori_gt.shape[-1]):
            temp_gt = ori_gt[:,:,:,j]
            xaxis = ori_gt.shape[0]
            yaxis = ori_gt.shape[1]
            zaxis = ori_gt.shape[2]
            
            # flip
            if args.flip == 'True':
                temp_gt = np.flip(temp_gt, flip_order)
            
            # shift
            file_shift = np.zeros([xaxis+2*abs(shift_range_x),yaxis+2*abs(shift_range_y),zaxis],dtype=float)
            # file_shift += ori_gt[0,0,0]
            for a in range(xaxis):
                for b in range(yaxis):
                    file_shift[a+abs(shift_range_x)+shift_range_x,b+abs(shift_range_y)+shift_range_y,:] = temp_gt[a,b,:]
            temp_gt = file_shift[abs(shift_range_x):xaxis+abs(shift_range_x),abs(shift_range_y):yaxis+abs(shift_range_y),:]        
            
            # zoom
            file_zoom = ndi.zoom(temp_gt, zoom_range, order=0, mode='nearest')
            xs = file_zoom.shape[0]
            ys = file_zoom.shape[1]
            zs = file_zoom.shape[2]
            if zoom_range >= 1:
                xh = (xs-xaxis)//2
                yh = (ys-yaxis)//2
                zh = (zs-zaxis)//2
                temp_gt = file_zoom[xh:xh+xaxis, yh:yh+yaxis, zh:zh+zaxis]
            elif zoom_range < 1:
                xh = (xaxis-xs)//2
                yh = (yaxis-ys)//2
                zh = (zaxis-zs)//2
                file_fix = np.zeros(shape=[xaxis,yaxis,zaxis])
                # file_fix += ori_img[0,0,0]
                file_fix[xh:xh+xs,yh:yh+ys,zh:zh+zs]=file_zoom
                temp_gt = file_fix
                del file_fix
            
            # rotate
            temp_gt = ndi.rotate(temp_gt, rotate_angle, axes=(0,1), order=0, mode='nearest', reshape=False)
            
            # combine
            new_gt[:,:,:,j] = temp_gt   
        
        # Categorize
        new_gt = np.expand_dims(new_gt, axis=0)
        y = np.zeros(new_gt.shape, dtype=float)
        for j in range(y.shape[-1]):
            y_temp = categorise_prediction(np.argmax(new_gt, axis = 4), j)
            y_temp = np.expand_dims(y_temp, axis = 4)
            y[0,:,:,:,j] = y_temp[:,:,:,0]
            del y_temp
        
        # saving the augment ground truth separately
        print('Generating augment Y no.' + str(i+1))
        if args.combine == 'True':
            Y_com = np.concatenate((Y_com, new_gt), axis=0)
        else:
            np.save(args.out + '/Y_' + name, new_gt.astype(bool)) #dir
        del new_gt
    
    # saving the combined file
    if args.combine == 'True':
        X_com = X_com[1:,:,:,:,:]
        Y_com = Y_com[1:,:,:,:,:].astype(bool)
        np.save(args.out + '/X.npy', X_com)
        np.save(args.out + '/Y.npy', Y_com)
    
    # saving the parameters
    with open(args.out + '/Parameter.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(param)
    
    
if __name__ == '__main__':
    main()