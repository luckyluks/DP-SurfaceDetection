import os
import random
import pandas as pd
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL 
import platform
import time

import pickle
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

# costum imports
from dataset import *
from networks import *
from CrossEntropy2d import *
import functions as func



# setup
num_epochs = 8
batch_size = 8
batch_size_eval = 8
learning_rate = 0.000001
model_id = "10"


def read_from_pickle(path):
    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects


# Set paths -- splits if validation folders are not defined!!
if platform.system()=="Windows":
    path_train_images = "C:\\Users\\lukas\\workspace\\data\\GrabCutGroundTruthDec10\\data_splitted_M-building\\Frames"
    path_train_targets = "C:\\Users\\lukas\\workspace\\data\\GrabCutGroundTruthDec10\\data_splitted_M-building\\trueFrames"
    path_val_images = "C:\\Users\\lukas\\workspace\\data\\GrabCutGroundTruthDec10\\data_splitted_M-building\\FramesVal"
    path_val_targets = "C:\\Users\\lukas\\workspace\\data\\GrabCutGroundTruthDec10\\data_splitted_M-building\\trueFramesVal"
elif platform.system()=="Linux":
    path_train_images = "/media/zed/Data/gtdata/data/Frames"
    path_train_targets = "/media/zed/Data/gtdata/data/trueFrames"
    path_val_images = None
    path_val_targets = None

if os.path.exists(path_train_images) and os.path.exists(path_train_targets):
    print("Using platform: {} (release: {})".format(platform.system(), platform.release()) )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Data paths valid: {}".format(True))
    print("Using separat validation folder: {}".format((path_val_images!=None) and (path_val_targets!=None)))

    # get all file names from directories
    image_paths = [os.path.join(path_train_images, image_item) for image_item in os.listdir(path_train_images)]
    target_paths = [os.path.join(path_train_targets, target_item) for target_item in os.listdir(path_train_targets)]
    dataset = MyDataset(image_paths, target_paths, train=True)

    if (path_val_images!=None) and (path_val_targets!=None):
        val_image_paths = [os.path.join(path_val_images, image_item) for image_item in os.listdir(path_val_images)]
        val_target_paths = [os.path.join(path_val_targets, target_item) for target_item in os.listdir(path_val_targets)]

        val_dataset = MyDataset(val_image_paths, val_target_paths, train=False)
        train_dataset = dataset
    else:        
        n_samples = len(dataset)
        n_train_samples = int(n_samples*0.8)
        n_val_samples = n_samples - n_train_samples
        train_dataset, val_dataset = random_split(dataset, [n_train_samples, n_val_samples])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_eval, shuffle=False)
    print("train_data_loader files:", len(train_loader.dataset))
    print("val_data_loader files:", len(val_loader.dataset))

else:
    # print('Dataset not available in:\n{}\n{}'.format(path_train_images,path_train_targets))
    raise NotImplementedError('ERROR: Dataset not available in:\n\t{}\n\t{}'.format(path_train_images,path_train_targets))


# network
model_dir = os.path.join(os.getcwd(),"neuralnetwork","model")
network =  U_Net(img_ch=3,output_ch=2)
model_dir_files = os.listdir(model_dir)
previous_model_checkpoints = [ string for string in model_dir_files if "model_"+model_id in string]

test = read_from_pickle("%s/epoch_losses_val.pkl" % model_dir)

# continue learning if model-id already used
epoch_offset = 0
if len(previous_model_checkpoints) > 0:
    epoch_numbers = [int(filename.split("_")[-1].split(".")[0]) for filename in previous_model_checkpoints]
    model_dir = os.path.join(os.getcwd(),"neuralnetwork","model")
    model_name = "model_" + str(model_id) + "_epoch_" + str(max(epoch_numbers))
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

print("START time:", str(time.strftime("%Y-%m-%d - %H:%M:%S")))

for epoch in range(num_epochs):
    print ("################################### NEW EPOCH : %d/%d (offset: %d)" % (epoch+1, num_epochs, epoch_offset))

    network.train() # (set in training mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label_imgs, _) in enumerate(tqdm(train_loader)):
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
    plt.savefig("{}/model_{}_epoch_losses_train.png".format(model_dir,model_id))
    plt.close(1)

    print ("####")

    ############################################################################
    # val:
    ############################################################################
    network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label_imgs, file_name) in enumerate(tqdm(val_loader)):
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = Variable(imgs).to(device) # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = label_imgs.squeeze(1)
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device) # (shape: (batch_size, img_h, img_w))

            outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

            # compute the loss:
            loss = loss_fn(outputs, label_imgs)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)
    with open("%s/epoch_losses_val.pkl" % model_dir, "wb") as file:
        pickle.dump(epoch_losses_val, file)
    print ("val loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_val, "k^")
    plt.plot(epoch_losses_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val loss per epoch")
    plt.savefig("{}/model_{}epoch_losses_val.png".format(model_dir,model_id))
    plt.close(1)


  

    # save the model weights to disk:
    checkpoint_path = model_dir + "/model_" + model_id +"_epoch_" + str(epoch_offset+epoch+1) + ".pth"
    torch.save(network.state_dict(), checkpoint_path)

print("END time:", str(time.strftime("%Y-%m-%d - %H:%M:%S")))