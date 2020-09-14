# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:16:10 2020

@author: Ching-Ting Kurt Lin
"""

def unet3d_report(X, y_pred, output_image, voxel, channel_order):
    
    """
    Augs:
        X: numpy
            The shape of the matrix will be [number, height, width, depth, channel=1].
        y_pred: numpy
            The shape of the matrix will be [number, height, width, depth, channel].        
        output_image: string
            Path to save the image.
        voxel: float
            Transform the matrix from resolution to real length(px3 to mm3) to get the real volume.
        channel_order: list
            The channel of LCRB, LGM, LWM, RCRB, RGM, RWM. The default is [3,4,1,7,8,5].
    """
    
    import numpy as np
    import cv2
    from skimage.transform import resize
    import matplotlib.pyplot as plt
    
    LGM = y_pred[:,:,:,:,channel_order[0]]
    LWM = y_pred[:,:,:,:,channel_order[1]]
    LCRB = y_pred[:,:,:,:,channel_order[2]]
    RGM = y_pred[:,:,:,:,channel_order[3]]
    RWM = y_pred[:,:,:,:,channel_order[4]]
    RCRB = y_pred[:,:,:,:,channel_order[5]]
    
    # Plot image in axial(1), coronal and sagittal
    GM = LGM + RGM
    WM = LWM + RWM
    CRB = LCRB + RCRB
    GM = GM[0,:,:,:]
    WM = WM[0,:,:,:]
    CRB = CRB[0,:,:,:]
    zaxis = int(X.shape[3] * 5 / 8)
    img1 = X[0,:,:,zaxis,0]
    GM1 = GM[:,:,zaxis].astype(bool)
    WM1 = WM[:,:,zaxis].astype(bool)
    CRB1 = CRB[:,:,zaxis].astype(bool)
    yaxis = int(X.shape[2] * 7 / 16)
    img2 = X[0,:,yaxis,:,0]
    GM2 = GM[:,yaxis,:]
    WM2 = WM[:,yaxis,:]
    CRB2 = CRB[:,yaxis,:]
    xaxis = int(X.shape[2] * 15 / 32)
    img3 = X[0,xaxis,:,:,0]
    GM3 = GM[xaxis,:,:]
    WM3 = WM[xaxis,:,:]
    CRB3 = CRB[xaxis,:,:]
    
    img1 = resize(img1, [256,256])
    img2 = resize(img2, [256,256])
    GM2 = resize(GM2, [256,256]).astype(bool)
    WM2 = resize(WM2, [256,256]).astype(bool)
    CRB2 = resize(CRB2, [256,256]).astype(bool)
    img3 = resize(img3, [256,256])
    GM3 = resize(GM3, [256,256]).astype(bool)
    WM3 = resize(WM3, [256,256]).astype(bool)
    CRB3 = resize(CRB3, [256,256]).astype(bool)
    
    img1 = ((img1-img1.min())/(img1.max()-img1.min())).astype('float32')
    img2 = ((img2-img2.min())/(img2.max()-img2.min())).astype('float32')
    img3 = ((img3-img3.min())/(img3.max()-img3.min())).astype('float32')
    
    # plot WM, GM, CRB area in rgb
    out1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
    out1[(GM1),0] = .6
    out1[(WM1),2] = .6
    out1[(CRB1),1] = .6
    out2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
    out2[(GM2),0] = .6
    out2[(WM2),2] = .6
    out2[(CRB2),1] = .6
    out3 = cv2.cvtColor(img3,cv2.COLOR_GRAY2RGB)
    out3[(GM3),0] = .6
    out3[(WM3),2] = .6
    out3[(CRB3),1] = .6
    
    # ori_img, true, pred comparison
    img_all = np.concatenate((out1, out2, out3), axis=1)
    
    # save the plot
    plt.imshow(img_all)
    plt.axis('off')
    plt.savefig(output_image, dpi = 300, bbox_inches = "tight")
    plt.close('all')
