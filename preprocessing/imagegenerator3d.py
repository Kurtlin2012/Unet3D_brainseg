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


def imagegenerator3d(ori, gt, output_folder, num = 50):
    
    """
    Input:
        ori: 5-D Numpy 
            Matrix including all the original MR images made by nii2npy. Default shape of matrix is [number, height, width, depth, channel = 1].
        gt: 5-D Numpy
            Matrix including all the ground truth made by nii2npy. Default shape of matrix is [number, height, width, depth, channel = 10].
        output_folder: String
            Path of folder for augmented datas.
        num: int
            The amount of augmented datas.
    
    Output:
        new_img: 5-D Numpy
            Augmented MRI data. The shape of matrix is [number=1, hetght, width, depth, channel=1].            
        new_gt: 5-D Numpy
            Augmented ground truth. The shape of matrix is [number=1, hetght, width, depth, channel].
        para: List
            List of all parameters for each augmented data, including the index of original image, shift factor, zoom factor and rotate factor.
    """
    
    import os
    import numpy as np
    import random
    import scipy.ndimage as ndi
    import xlsxwriter as xlw
    
    # load the numpy files
    X_ori = np.load(ori)
    Y_ori = np.load(gt)
    
    # list for parameters
    param = [['No.', 'Original File Index','Shift Factor X','Shift Factor Y','Zoom Factor','Rotate Angle']]
        
    for i in range(num):
        # parameters
        order=random.randint(0,X_ori.shape[0]-1)
        flip_axis=None
        shift_range_x=random.randint(-3,3)
        shift_range_y=random.randint(-3,5)
        zoom_range=random.uniform(0.9,1.1)
        rotate_angle=random.uniform(-5,5)
        
        # read original image
        ori_img = X_ori[order,:,:,:,0]
        new_img = ori_img
        xaxis = ori_img.shape[0]
        yaxis = ori_img.shape[1]
        zaxis = ori_img.shape[2]
        
        # flip
        if flip_axis != None:
            new_img = np.flip(new_img, flip_axis)
        
        # shift
        file_shift = np.zeros([xaxis+2*abs(shift_range_x),yaxis+2*abs(shift_range_y),zaxis],dtype=float)
        file_shift += ori_img[0,0,0]
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
            file_fix += ori_img[0,0,0]
            file_fix[xh:xh+xs,yh:yh+ys,zh:zh+zs]=file_zoom
            new_img = file_fix
            del file_fix
        
        # rotate
        new_img = ndi.rotate(new_img, rotate_angle, axes=(0,1), order=0, mode='nearest', reshape=False)
    
        # normalize and save
        new_img = (new_img-new_img.mean())/new_img.std()
        new_img = np.expand_dims(new_img, axis = 0)
        new_img = np.expand_dims(new_img, axis = 4)
        param.append([i+1, order+1, shift_range_x, shift_range_y, zoom_range, rotate_angle])
        print('Saving augment X no.' + str(i+1))
        
        # save the augment image separately
        np.save(output_folder + '/X_' + str(i+1) + '.npy', new_img) 
        del new_img
        
        # read original ground truth
        ori_gt = Y_ori[order,:,:,:,:]
        new_gt = np.zeros(ori_gt.shape)
        for j in range(ori_gt.shape[-1]):
            temp_gt = ori_gt[:,:,:,j]
            xaxis = ori_gt.shape[0]
            yaxis = ori_gt.shape[1]
            zaxis = ori_gt.shape[2]
            
            # flip
            if flip_axis != None:
                temp_gt = np.flip(temp_gt, flip_axis)
            
            # shift
            file_shift = np.zeros([xaxis+2*abs(shift_range_x),yaxis+2*abs(shift_range_y),zaxis],dtype=float)
            file_shift += ori_gt[0,0,0]
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
                file_fix += ori_img[0,0,0]
                file_fix[xh:xh+xs,yh:yh+ys,zh:zh+zs]=file_zoom
                temp_gt = file_fix
                del file_fix
            
            # rotate
            temp_gt = ndi.rotate(temp_gt, rotate_angle, axes=(0,1), order=0, mode='nearest', reshape=False)
            
            # combine
            new_gt[:,:,:,j] = temp_gt   
        
        
        new_gt = np.round((new_gt-new_gt.min()/(new_gt.max()-new_gt.min()))).astype(bool)
        new_gt = np.expand_dims(new_gt, axis=0)
        
        # saving the augment ground truth separately
        print('Saving augment Y no.' + str(i+1))
        np.save(output_folder + '/Y_' + str(i+1) + '.npy', new_gt) #dir
        del new_gt
    
    # saving the parameters
    os.chdir(output_folder) #dir
    wb = xlw.Workbook('param.xlsx')
    ws = wb.add_worksheet()
    for row, item in enumerate(param):
        ws.write_row(row, 0, item)
    wb.close()
