import os
import random
import pandas as pd
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL


# from tqdm import tqdm_notebook, tnrange
# from itertools import chain
# from skimage.io import imread, imshow, concatenate_images
# from skimage.transform import resize
# from skimage.morphology import label
# from sklearn.model_selection import train_test_split

# import tensorflow as tf

# from keras.models import Model, load_model
# from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
# from keras.layers.core import Lambda, RepeatVector, Reshape
# from keras.layers.convolutional import Conv2D, Conv2DTranspose
# from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
# from keras.layers.merge import concatenate, add
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.optimizers import Adam
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import pickle
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from dataset import *
from networks import *
from CrossEntropy2d import *
import functions as func



# Set some parameters
im_width = 128
im_height = 128
border = 5
path_train =  r'data'
path_test = '../input/test/'

# path_root = "C:\\Users\\lukas\\workspace\\data\\raw_data"
# path_train = "C:\\Users\\lukas\\workspace\\data\\raw_data\\Frames"
# path_test = "C:\\Users\\lukas\\workspace\\data\\raw_data\\trueFrames"

path_train = "/media/zed/Data/gtdata/data/Frames"
path_test = "/media/zed/Data/gtdata/data/trueFrames"



# hymenoptera_dataset = datasets.ImageFolder(root=path_root)
# dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
#                                              batch_size=4, shuffle=True,
#                                              num_workers=4)

# print(dataset_loader.dataset.shape)


# setup
num_epochs = 10
num_epochs_eval = 1
batch_size = 10
batch_size_eval = 2
learning_rate = 0.0001
model_id = "6"

image_paths = os.listdir(path_train)
target_paths = os.listdir(path_test)

image_paths = [os.path.join(path_train, image_item) for image_item in image_paths]

target_paths = [os.path.join(path_test, target_item) for target_item in target_paths]
dataset = MyDataset(image_paths, target_paths, train=True)

# n_samples = len(dataset)
# n_train_samples = int(n_samples*0.7)
# n_val_samples = n_samples - n_train_samples
# train_dataset, val_dataset = random_split(dataset, [n_train_samples, n_val_samples])
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size_eval)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)



print("train_data_loader:", len(train_loader.dataset))
# print("val_data_loader:", len(val_loader.dataset))
# import matplotlib.pyplot as plt
print("done")

# for x,y in train_data_loader:
#     plt.imshow(x[0].numpy().transpose((1, 2, 0)), cmap='gray');
#     plt.show()
#     plt.imshow(y[0].numpy().transpose((1, 2, 0)), cmap='gray');
#     plt.show()
#     break


# network
model_dir = os.path.join(os.getcwd(),"neuralnetwork","model")
network =  U_Net(img_ch=3,output_ch=2)
model_dir_files = os.listdir(model_dir)
previous_model_checkpoints = [ string for string in model_dir_files if "model_"+model_id in string]
# continue learning if model-id already used
epoch_offset = 0
if len(previous_model_checkpoints) > 0:
    epoch_numbers = [int(filename.split("_")[-1].split(".")[0]) for filename in previous_model_checkpoints]
    model_dir = os.path.join(os.getcwd(),"neuralnetwork","model")
    model_file_name = "model_" + str(model_id) + "_epoch_" + str(max(epoch_numbers)) + ".pth"
    state = torch.load(os.path.join(model_dir, model_file_name))
    network.load_state_dict(state)
    epoch_offset = max(epoch_numbers)
    print("loaded previous checkpoint: ", model_file_name)

network.to(device)


# loss function
loss_fn = nn.CrossEntropyLoss()

# params = add_weight_decay(network, l2_value=0.0001)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

epoch_losses_train = []
epoch_losses_val = []
for epoch in range(num_epochs):
    print ("################################### NEW EPOCH : %d/%d (offset: %d)" % (epoch+1, num_epochs, epoch_offset))

    ############################################################################
    # train:
    ############################################################################
    network.train() # (set in training mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label_imgs, _) in enumerate(tqdm(train_loader)):
        #current_time = time.time()

        imgs = Variable(imgs).to(device) # (shape: (batch_size, 3, img_h, img_w))
        label_imgs = label_imgs.squeeze(1)
        label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device) # (shape: (batch_size, img_h, img_w))

        outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

        # print("img: \t", imgs.shape)
        # print("tar: \t", label_imgs.shape)
        # print("out: \t", outputs.shape)

        # # test = label_imgs[0].cpu().detach().numpy()
        # # np.unique(test)
        # # plt.imshow(, cv2.imwrite("test.png", test.transpose(1,2,0)*255))

        # # outputs = outputs.squeeze(1)

        # print("img: \t", imgs.shape)
        # print("tar: \t", label_imgs.shape)
        # print("out: \t", outputs.shape)

        # label_imgs=label_imgs.argmax(1)

        # print(outputs.shape)
        # print(label_imgs.shape)

        # compute the loss:
        loss = loss_fn(outputs, label_imgs)
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

        # optimization step:
        optimizer.zero_grad() # (reset gradients)
        loss.backward() # (compute gradients)
        optimizer.step() # (perform optimization step)

        # #print (time.time() - current_time)
        # for step, (imgs, label_imgs) in enumerate(val_loader):
        #     with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        #         imgs = Variable(imgs).to(device) # (shape: (batch_size, 3, img_h, img_w))
        #         label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device) # (shape: (batch_size, img_h, img_w))

        #         outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

        #         # compute the loss:
        #         # loss = loss_fn(outputs, label_imgs)
        #         # loss_value = loss.data.cpu().numpy()
        #         # batch_losses.append(loss_value)

        #         # f, axarr = plt.subplots(imgs.shape[0],3)

        #         # for idx in range(imgs.shape[0]):
        #         #     image = imgs[idx].cpu().detach().numpy().transpose((1,2,0))
        #         #     target = label_imgs[idx].cpu().detach().numpy().transpose((1,2,0))
        #         #     output = outputs[idx].cpu().detach().numpy().transpose((1,2,0))
        #         #     axarr[idx,0].imshow(image)
        #         #     axarr[idx,1].imshow(target)
        #         #     axarr[idx,2].imshow(output)
        #         # plt.show()

    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)
    with open("%s/epoch_losses_train.pkl" % model_dir, "wb") as file:
        pickle.dump(epoch_losses_train, file)
    print ("train loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_train, "k^")
    plt.plot(epoch_losses_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train loss per epoch")
    plt.savefig("%s/epoch_losses_train.png" % model_dir)
    plt.close(1)

    # print ("####")

    # ############################################################################
    # # val:
    # ############################################################################
    # network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
    # batch_losses = []
    # for step, (imgs, label_imgs, img_ids) in enumerate(val_loader):
    #     with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
    #         imgs = Variable(imgs).to(device) # (shape: (batch_size, 3, img_h, img_w))
    #         label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device) # (shape: (batch_size, img_h, img_w))

    #         outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

    #         # compute the loss:
    #         loss = loss_fn(outputs, label_imgs)
    #         loss_value = loss.data.cpu().numpy()
    #         batch_losses.append(loss_value)

    # epoch_loss = np.mean(batch_losses)
    # epoch_losses_val.append(epoch_loss)
    # with open("%s/epoch_losses_val.pkl" % model_dir, "wb") as file:
    #     pickle.dump(epoch_losses_val, file)
    # print ("val loss: %g" % epoch_loss)
    # plt.figure(1)
    # plt.plot(epoch_losses_val, "k^")
    # plt.plot(epoch_losses_val, "k")
    # plt.ylabel("loss")
    # plt.xlabel("epoch")
    # plt.title("val loss per epoch")
    # plt.savefig("%s/epoch_losses_val.png" % model_dir)
    # plt.close(1)

    # save the model weights to disk:
    checkpoint_path = model_dir + "/model_" + model_id +"_epoch_" + str(epoch_offset+epoch+1) + ".pth"
    torch.save(network.state_dict(), checkpoint_path)


# iou_sum = 0
# num_steps = 0
# for step, (imgs, label_imgs) in enumerate(val_loader):
#         with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
#             imgs = Variable(imgs).to(device) # (shape: (batch_size, 3, img_h, img_w))
#             label_imgs = label_imgs.squeeze(1)
#             label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device) # (shape: (batch_size, img_h, img_w))
#             outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))


#             f, axarr = plt.subplots(imgs.shape[0],3)

#             for idx in range(imgs.shape[0]):
#                 image = imgs[idx].detach().numpy().transpose((1,2,0))
#                 target = label_imgs[idx].detach().numpy().transpose((1,2,0))
#                 output = outputs[idx].detach().numpy().transpose((1,2,0))
#                 axarr[idx,0].imshow(image)
#                 axarr[idx,1].imshow(target)
#                 axarr[idx,2].imshow(output)
#             plt.show()

#             iou_sum += func.intersectionOverUnion(outputs,label_imgs)
#             num_steps = step
# print('total IoU = {}'.format(iou_sum/num_steps))


#     batch_size = 1
# c, h, w = 3, 10, 10
# nb_classes = 5

# x = torch.randn(batch_size, c, h, w)
# target = torch.empty(batch_size, h, w, dtype=torch.long).random_(nb_classes)

# model = nn.Sequential(
#     nn.Conv2d(c, 6, 3, 1, 1),
#     nn.ReLU(),
#     nn.Conv2d(6, nb_classes, 3, 1, 1)
# )

# criterion = nn.CrossEntropyLoss()

# output = model(x)
# loss = criterion(output, target)
# loss.backward()