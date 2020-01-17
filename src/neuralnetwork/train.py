import os
import random
import pandas as pd
import numpy as np
import cv2
import glob


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


# Set some parameters
# im_width = 128
# im_height = 128
border = 5
path_train =  r'data'
path_test = '../input/test/'

path_train = "C:\\Users\\lukas\\workspace\\data\\raw_data"
path_test = "C:\\Users\\lukas\\workspace\\data\\raw_data"


# Get and resize train images and masks
def get_data(path, train=True):
    ids = next(os.walk(path + "/Frames"))[2]
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    # for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
    for n, id_ in enumerate(ids):
        # Load images
        img = load_img(path + '/Frames/' + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/trueFrames/' + id_, grayscale=True))
            mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X
    
X, y = get_data(path_train, train=True)

# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)


print('TRAIN: Min: {}\nMax: {}'.format(X_train.min(), X_train.max()))
print('VALID: Min: {}\nMax: {}'.format(X_valid.min(), X_valid.max()))
print('TRAIN: Min: {}\nMax: {}'.format(y_train.min(), y_train.max()))
print('VALID: Min: {}\nMax: {}'.format(y_valid.min(), y_valid.max()))


print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(y_valid.shape)

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

dataset = TensorDataset(torch.from_numpy(X_train).to(device), torch.from_numpy(y_train).to(device))
val_dataset = TensorDataset(torch.from_numpy(X_valid).to(device), torch.from_numpy(y_valid).to(device))
train_data_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

print("train_data_loader:", len(train_data_loader.dataset))
print("val_data_loader:", len(val_data_loader.dataset))
# import matplotlib.pyplot as plt
print("done")
# plt.imshow(X_train[10], cmap='gray');
# plt.imshow(X_valid[10], cmap='gray');
# plt.imshow(y_train[10], cmap='gray');
# plt.imshow(y_valid[10], cmap='gray');

# # Check if training data looks all right
# ix = random.randint(0, len(X_train))
# has_mask = y_train[ix].max() > 0

# # train_dataset = CustomDataset(train_image_paths, train_mask_paths, train=True)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)

# # test_dataset = CustomDataset(test_image_paths, test_mask_paths, train=False)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1)