# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:07:49 2019

@author: onthe
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

im_width = 128
im_height = 128
border = 5

path_train =  r'C:\Users\onthe\Downloads\study\6-design_project\Kaggle_salt\tgs-salt-identification-challenge\data_aug'
path_test = '../input/test/'

#%%
# Get and resize train images and masks
def get_data(path, train=True):
    ids = next(os.walk(path + "/images"))[2]
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + '/images/' + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_height, im_width, 1), mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_, grayscale=True))
            mask = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255

    if train:
        return X, y
    else:
        return X
    print('Done!')
    
X, y = get_data(path_train, train=True)

#%%

# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)

#%%
# Check if training data looks all right
ix = random.randint(0, len(X_train))
has_mask = y_train[ix].max() > 0

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

#ax[0].imshow(X_train[ix, ..., 0], cmap='seismic', interpolation='bilinear')
ax[0].imshow(X_train[ix, ..., 0], interpolation='bilinear')
if has_mask:
    ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
ax[0].set_title('Seismic')

ax[1].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
ax[1].set_title('GroundTruth');


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
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

#%%
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-unet-1230-aug.h5', verbose=1, save_best_only=True, save_weights_only=True)
]


results = model.fit(X_train, y_train, batch_size=64, epochs=10, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))

#%%
plt.figure(figsize=(8, 8))

plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();

plt.savefig('learning_curve.png')
#%%


#%%
# Load model
model.load_weights('model-unet-1230-aug.h5')
#model.load_weights('model-unet-1210.h5')

#%%

# Evaluate on validation set (this must be equals to the best log_loss)
model.evaluate(X_valid, y_valid, verbose=1)

#%%

# Predict on train, val and test
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)

#%%
Threshold = 0.4
# Threshold predictions
preds_train_t = (preds_train > Threshold).astype(np.uint8)
preds_val_t = (preds_val > Threshold).astype(np.uint8)

#%%
'''
def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(40, 20))
    ax[0].imshow(X[ix, ..., 0])
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Original image')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Ground truth')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('My Predicted')
    
    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5]) 
    ax[3].set_title('My Predicted binary');
'''
#%%  
def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 3, figsize=(30, 30))
    ax[0].imshow(X[ix].squeeze(),cmap= 'gray')    
    ax[0].set_title('Original image')

    ax[1].imshow(y[ix].squeeze(),alpha = 0.9)  
    ax[1].set_title('Ground truth')
    
    ax[2].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1,alpha = 0.9)
    ax[2].set_title('My Predicted binary'); 

#%%
    
# Check if training data looks all right
plot_sample(X_train, y_train, preds_train, preds_train_t, ix=29)

#%%
# Check if valid data looks all right
for i in range(1):
    plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=i)
    
#%%
   

os.makedirs(r'C:\Users\onthe\Downloads\study\6-design_project\Kaggle_salt\tgs-salt-identification-challenge\GrabCutGroundTruthDec10\out//', exist_ok=True)
 
def save_samples(X, y, preds, binary_preds):
 
    for ix in range(len(preds)):
 
        originalImage = X[ix, ..., 0]
 
        groundTruth = y[ix].squeeze()
 
        prediction = preds[ix].squeeze()
 
        binaryPred = binary_preds[ix].squeeze()
 
        c1 = np.hstack((originalImage,groundTruth))
        c2 = np.hstack((binaryPred,prediction))
        comb = np.vstack((c1,c2))
        comb = np.round(comb*255)
        comb = comb.astype(np.uint8)
 
 
        cv2.imwrite(r'C:\Users\onthe\Downloads\study\6-design_project\Kaggle_salt\tgs-salt-identification-challenge\GrabCutGroundTruthDec10\out//'+str(ix+1)+'.jpg',comb)
 
save_samples(X_valid, y_valid, preds_val, preds_val_t)    

        
#%%
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(r"C:\Users\onthe\Downloads\study\6-design_project\GIT\DP-SurfaceDetection-lukas\recordings\object_detection_30s_20191126-161118.mp4")

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
        preds = model.predict(frame, verbose=1)
        preds_t = (preds > 0.9).astype(np.uint8)*255 
        '''
        print('Frame',np.shape(Frame))
        print('frame',np.shape(frame))
        print('preds',np.shape(preds))
        print('pred_t',np.shape(preds_t))
        '''
       # Frame = resize(Frame, ( 240*1, 320*1, 1), mode='constant', preserve_range=False)
        preds = resize(preds, (1, 180*1, 320*1, 1), mode='constant', preserve_range=False)
        
        preds_t = resize(preds_t, (1, 180*2, 320*2, 1), mode='constant', preserve_range=True)
        #print('predst',np.shape(preds_t))
        
        cv2.imshow('Frame', Frame)
        cv2.imshow('prediction', preds.squeeze())
        cv2.imshow('prediction binary', preds_t.squeeze())
        

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()