import os
import random
import pandas as pd
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL


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

path_train = "/media/zed/Data/gtdata/dataTest/Frames"
path_test = "/media/zed/Data/gtdata/dataTest/trueFrames"
output_folder = "/media/zed/Data/gtdata/dataTest/predictions"

path_train = "C:\\Users\\lukas\\workspace\\data\\GrabCutGroundTruthDec10\\data_splitted_M-building\\FramesVal"
path_test = "C:\\Users\\lukas\\workspace\\data\\GrabCutGroundTruthDec10\\data_splitted_M-building\\trueFramesVal"
output_folder = "C:\\Users\\lukas\\workspace\\data\\GrabCutGroundTruthDec10\\data_splitted_M-building\\predictions"



# setup
batch_size = 1
model_id = 0

image_paths = os.listdir(path_train)
target_paths = os.listdir(path_test)

image_paths = [os.path.join(path_train, image_item) for image_item in image_paths]
target_paths = [os.path.join(path_test, target_item) for target_item in target_paths]

# image_paths = ['./data/0.png', './data/1.png']
# target_paths = ['./target/0.png', './target/1.png']
dataset = MyDataset(image_paths, target_paths, train=False)

# n_samples = len(dataset)
# n_train_samples = int(n_samples*0.7)
# n_val_samples = n_samples - n_train_samples
# train_dataset, val_dataset = random_split(dataset, [n_train_samples, n_val_samples])
# train_loader = DataLoader(train_dataset, batch_size=num_epochs, shuffle=True)
val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)
print("val_data_loader files:", len(val_loader.dataset))

# network
model_dir = os.path.join(os.getcwd(),"neuralnetwork","model")
network =  U_Net(img_ch=3,output_ch=2)
model_dir_files = os.listdir(model_dir)

if model_id == 0:
    all_model_checkpoints = [ string for string in model_dir_files if "model_" in string]
    previous_model_ids = [int(filename.split("_")[1]) for filename in all_model_checkpoints]
    model_id = max(previous_model_ids)
model_id_checkpoints = [ string for string in model_dir_files if "model_"+str(model_id) in string]
epoch_numbers = [int(filename.split("_")[-1].split(".")[0]) for filename in model_id_checkpoints]
model_dir = os.path.join(os.getcwd(),"neuralnetwork","model")
model_file_name = "model_" + str(model_id) + "_epoch_" + str(max(epoch_numbers)) + ".pth"
state = torch.load(os.path.join(model_dir, model_file_name))
network.load_state_dict(state)
epoch_offset = max(epoch_numbers)
print("loaded previous checkpoint ( {}): {}".format(epoch_offset, model_file_name))

network.to(device)
network.eval()


iou_sum = 0
num_steps = 0
for step, (imgs, label_imgs, file_name) in enumerate(tqdm(val_loader)):
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = Variable(imgs).to(device) # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = label_imgs.squeeze(1)
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device) # (shape: (batch_size, img_h, img_w))
            outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))
            
            outputs = outputs.squeeze(1)

            # get to device
            imgs = imgs.cpu()
            label_imgs = label_imgs.cpu()
            outputs = outputs.cpu()

            outputs = torch.argmax(outputs, dim=1)

            # (np.unique(outputs.cpu()))

            # print("img: \t", imgs.shape)
            # print("tar: \t", label_imgs.shape)
            # print("out: \t", outputs.shape)

            # f, axarr = plt.subplots(imgs.shape[0],3)

            for idx in range(imgs.shape[0]):
                image = imgs[idx].cpu().detach().numpy().transpose((1,2,0))
                target = label_imgs[idx].cpu().detach().numpy().transpose((0,1))
                output = outputs[idx].cpu().detach().numpy().transpose((0,1))

                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                images = np.hstack((image_gray*255, target*255, output*255))
                

                cv2.imwrite(os.path.join(output_folder, file_name[0]),images)
               
                # img = cv2.imread(os.path.join(output_folder, file_name[0]),0)
                # cv2.namedWindow('frames', cv2.WINDOW_AUTOSIZE)
                # cv2.imshow('frames', img)
                # cv2.waitKey(0)


                # axarr[idx,0].imshow(Image.fromarray(image))
                # axarr[idx,1].imshow(Image.fromarray(output*255))
                # axarr[idx,2].imshow(output)
            # plt.show()

            iou_sum += func.intersectionOverUnion(outputs,label_imgs)
            num_steps = step
print('total for IoU = {}'.format(iou_sum/num_steps))