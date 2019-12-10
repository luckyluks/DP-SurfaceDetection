import os
import random
import pandas as pd
import numpy as np
import cv2
import Functions as func


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


im_width = 640
im_height = 480
border = 5

#create network output folder
os.makedirs('data/outputUnetTF', exist_ok=True)


path_train =  r'data'


np.random.seed(1337)
order = np.random.randint(1,7564,size=500)


def get_data(path, train=True):
    # ids = next(os.walk(path + "/Frames"))[2]

    X = np.zeros((len(order), im_height, im_width, 1), dtype=np.float32)
    y = np.zeros((len(order), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in enumerate(order):
        # Load images
        img = load_img(path + '/Frames/frame' + str(id_) + '.jpg', grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_height, im_width, 1), mode='constant', preserve_range=True)  #EDIT

        # Load masks
        mask = img_to_array(load_img(path + '/trueFrames/frame' + str(id_) + '.jpg', grayscale=True))
        mask = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        y[n] = mask / 255
    print('Done!')
    return X,y

X_valid, y_valid  = get_data(path_train, train=False)

# Split train and valid
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=2018)

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


# %%
def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
print('model loaded')

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

#%%
# Load best model
model.load_weights('model-unet.h5')
print('weights loaded')
#%%
# Evaluate on validation set (this must be equals to the best log_loss)
# model.evaluate(X_valid, y_valid, verbose=1)

#%%
print('predicting')

iousum = 0
for imn in range(500):

    print(imn)

    index = order[imn]

    groundTruth = y_valid[imn]
    groundTruth = groundTruth.squeeze()

    input = X_valid[imn]
    input = input[np.newaxis, :, :, :]
    prediction = model.predict(input, verbose=1)

    binaryPred = (prediction > 0.5).astype(np.uint8)

# originalImage = X[ix, ..., 0]
        #
        # groundTruth = y[ix].squeeze()

    prediction = prediction.squeeze()

    binaryPred = binaryPred.squeeze()

    # c1 = np.hstack((originalImage,groundTruth))
    c2 = np.hstack((prediction,binaryPred))
    # comb = np.vstack((c1,c2))
    comb = np.round(c2*255)
    comb = comb.astype(np.uint8)

    cv2.imwrite('data/outputUnetTF/frame'+str(index)+'.jpg',comb)

    groundTruth = groundTruth.astype(np.uint8)
    iousum += func.intersectionOverUnion(binaryPred, groundTruth)

print('total IoU = {}'.format(iousum/500))

