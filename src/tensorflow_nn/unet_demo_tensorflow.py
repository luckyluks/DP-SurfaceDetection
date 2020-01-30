# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:07:49 2019

@author: zy
"""
import cv2
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#%matplotlib inline

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#%%
# Set some parameters

im_width = 320
im_height = 240

#path_train =  r'C:\Users\onthe\Downloads\study\6-design_project\Kaggle_salt\tgs-salt-identification-challenge\GrabCutGroundTruthDec10'

#%%

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

    
input_img = Input((im_height, im_width, 1), name='img')
model1 = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model2 = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model3 = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

#%%
# Load model
model1.load_weights('model-unet-1215.h5')
model2.load_weights('model-unet.h5')
model3.load_weights('model-unet-1230-aug.h5')
#%%
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(r"C:\Users\onthe\Downloads\study\6-design_project\GIT\DP-SurfaceDetection-lukas\recordings\object_detection_30s_20191126-161118.mp4")
Threshold = 0.5
while 1:
    ret, frame = cap.read()
    if frame is None:
        cv2.waitKey(0)
        break
    else:
        
        frame = frame[61:420,:]
        Frame = frame
        frame = resize(frame, ( im_height, im_width, 1), mode='constant', preserve_range=True)
        frame = img_to_array(frame)
        frame = frame[None]/255
        preds1 = model1.predict(frame, verbose=1)
        preds_t1 = (preds1 > Threshold).astype(np.uint8)*255 
        preds2 = model2.predict(frame, verbose=1)
        preds_t2 = (preds2 > Threshold).astype(np.uint8)*255 
        preds3 = model3.predict(frame, verbose=1)
        preds_t3 = (preds3 > Threshold).astype(np.uint8)*255 

       # Frame = resize(Frame, ( 240*1, 320*1, 1), mode='constant', preserve_range=False)
        preds1 = resize(preds1, (1, 180*1, 320*1, 1), mode='constant', preserve_range=False)
        preds2 = resize(preds2, (1, 180*1, 320*1, 1), mode='constant', preserve_range=False)
        preds_t1 = resize(preds_t1, (1, 180*2, 320*2, 1), mode='constant', preserve_range=True)
        preds_t2 = resize(preds_t2, (1, 180*2, 320*2, 1), mode='constant', preserve_range=True)
        preds3 = resize(preds1, (1, 180*1, 320*1, 1), mode='constant', preserve_range=True)
        preds_t3 = resize(preds_t3, (1, 180*2, 320*2, 1), mode='constant', preserve_range=True)

        
        cv2.imshow('Frame', Frame)
        cv2.imshow('prediction1', preds1.squeeze())
        cv2.imshow('prediction binary1', preds_t1.squeeze())
        cv2.imshow('prediction2', preds2.squeeze())
        cv2.imshow('prediction binary2', preds_t2.squeeze())
        cv2.imshow('prediction3', preds3.squeeze())
        cv2.imshow('prediction binary3', preds_t3.squeeze())
        

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()