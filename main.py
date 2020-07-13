# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:24:51 2020

@author: Ching-Ting Kurt Lin
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.callbacks import History
from model.unet3d import unet3d


def unet3d_train(X_dir, Y_dir, output_folder, pretrained_weights=None):
    X = np.load(X_dir)
    Y = np.load(Y_dir)
    premodel = pretrained_weights
    unet = unet3d(input_shape=[X.shape[1], X.shape[2], X.shape[3], X.shape[4]], channels=Y.shape[-1], pretrained_weights=premodel)
    
    # callbacks
    history = History()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-20, verbose=1)
    early_stopping = EarlyStopping(monitor='val_dice_coef', patience=15, verbose=2, mode='max')
    model_checkpoint = ModelCheckpoint(output_folder + '/model-{epoch:05d}-{val_dice_coef:.5f}.h5', monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')
    
    # training
    unet.fit(X, Y, batch_size=1, epochs=50, verbose=1, callbacks=[history, model_checkpoint, reduce_lr, early_stopping], validation_split=0.1, shuffle=True)
    
    # save a dictionary into a pickle file.
    hist = {'dice_coef':history.history['dice_coef'],'val_dice_coef':history.history['val_dice_coef'],'loss':history.history['loss'],'val_loss':history.history['val_loss']}
    output = open(output_folder +'/history.pickle', "wb")
    pickle.dump(hist, output)
    
    # save plot(dice_coef, IoU, loss)
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model dice_coef'), plt.ylabel('dice_coef'), plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(output_folder + '/dice_coef.png')
    plt.show()
    plt.close('all')
    
    plt.plot(history.history['IoU'])
    plt.plot(history.history['val_IoU'])
    plt.title('model IoU'), plt.ylabel('IoU'), plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(output_folder + '/IoU.png')
    plt.show()
    plt.close('all')
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss'), plt.ylabel('loss'), plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(output_folder + '/loss.png')
    plt.show()
    plt.close('all')