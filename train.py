# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:24:51 2020

@author: Ching-Ting Kurt Lin
"""

import os
import argparse
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, History
from model.unet3d import unet_model

"""
Output:
    Trained weights: .h5 file
        This model will check the dice coefficient in each epoch, if the coefficient grows, the weight will be kept.
    Line charts: .png file
        At the end of training, the line charts of dice coefficient, loss and IoU will be saved, including results of training data and validation data.
    History: .csv file
        A list recording the dice coefficient, IoU and dice loss of each epoch.
"""

def parse_args(args):
    parser = argparse.ArgumentParser(description='Unet3D training')
    parser.add_argument('--train', type = str, help = 'File path of the training data (5-D numpy matrix).')
    parser.add_argument('--target', type = str, help = 'File path of the label/ground truth (5-D numpy matrix).')
    parser.add_argument('--out', type = str, help = 'Folder path to save the trained weights and the line charts of dice coefficient, loss and IoU.')
    parser.add_argument('--weight', type = str, default = None, help = 'File path of the pretrained weights(h5 file).')
    parser.add_argument('--bz', type = int, default = 1, help = 'Batch size of the training.')
    parser.add_argument('--epochs', type = int, default = 50, help = 'Epoch of the training.')
    parser.add_argument('--early', type = str, default = False, help = 'Enable/Disable the EarlyStopping function (True/False).')
    parser.add_argument('--init_f', type = int, default = 32, help = 'Number of the filter in the first encoder.')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'Set the learning rate of the model.')
    return parser.parse_args(args)


def main(args = None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    if not os.path.exists(args.out):
        os.mkdir(args.out)
    
    X = np.load(args.train)
    Y = np.load(args.target)
        
    input_shape = [X.shape[1], X.shape[2], X.shape[3], X.shape[4]]
    channels = Y.shape[-1]
    
    unet = unet_model(input_shape, channels, args.weight, args.lr, args.init_f)
    model = unet.model
    model.summary()
    
    # Training
    history = History()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-20, verbose=1)
    if not os.path.exists(args.out + '/Model'):
        os.mkdir(args.out + '/Model')
    model_checkpoint = ModelCheckpoint(args.out + '/Model/model-{epoch:05d}-{val_dice_coef:.5f}.h5', monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')
    if args.early == 'True':
        early_stopping = EarlyStopping(monitor='val_dice_coef', patience=15, verbose=2, mode='max')
        model.fit(X, Y, batch_size = args.bz, epochs = args.epochs, verbose=1, callbacks=[history, model_checkpoint, reduce_lr, early_stopping], validation_split=0.1, shuffle=True)
    else:
        model.fit(X, Y, batch_size = args.bz, epochs = args.epochs, verbose=1, callbacks=[history, model_checkpoint, reduce_lr], validation_split=0.1, shuffle=True)
     
        
    # Save a dictionary into a pickle file.
    hist = {'dice_coef':history.history['dice_coef'], 'val_dice_coef':history.history['val_dice_coef'], 'loss':history.history['loss'], 'val_loss':history.history['val_loss'], 'IoU':history.history['IoU'], 'val_IoU':history.history['val_IoU']}
    with open(args.out + '/History.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(hist)
    
    # save plot(dice_coef, IoU, loss)
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model dice_coef'), plt.ylabel('dice_coef'), plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(args.out + '/dice_coef.png')
    plt.show()
    plt.close('all')
    
    plt.plot(history.history['IoU'])
    plt.plot(history.history['val_IoU'])
    plt.title('model IoU'), plt.ylabel('IoU'), plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(args.out + '/IoU.png')
    plt.show()
    plt.close('all')
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss'), plt.ylabel('loss'), plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(args.out + '/loss.png')
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
