###################################################################################################
# author: Lukas Rauh
# description:  script to run evaluation on a test set and calculate IoU scores
###################################################################################################
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
from modeling.unet import *
import functions as func


# setup - choose model and paths for validation frames and groundtruth frames
model_id = 20
path_train = "C:\\Users\\lukas\\workspace\\data\\GrabCutGroundTruthDec10\\data_splitted\\FramesVal"
path_test = "C:\\Users\\lukas\\workspace\\data\\GrabCutGroundTruthDec10\\data_splitted\\trueFramesVal"
output_folder = "C:\\Users\\lukas\\workspace\\data\\GrabCutGroundTruthDec10\\data_splitted_M-building\\predictions"



# load data
batch_size = 1
image_paths = os.listdir(path_train)
target_paths = os.listdir(path_test)
image_paths = [os.path.join(path_train, image_item) for image_item in image_paths]
target_paths = [os.path.join(path_test, target_item) for target_item in target_paths]

dataset = MyDataset(image_paths, target_paths, train=False, resize_size=(640, 480))
val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)
print("val_data_loader files:", len(val_loader.dataset))

# load network
model_dir = os.path.join(os.getcwd(),"src","pytorch_nn","model")
model_dir_files = os.listdir(model_dir)

# choose model "automatically" - see try/except below
network =  U_Net(img_ch=3,output_ch=2)

if model_id == 0:
    all_model_checkpoints = [ string for string in model_dir_files if "model_" in string] 
    previous_model_ids = [int(filename.split("_")[1]) for filename in all_model_checkpoints]
    model_id = max(previous_model_ids)
model_id_checkpoints = [ string for string in model_dir_files if "model_"+str(model_id) in string]
epoch_numbers = [int(filename.split("_")[-1].split(".")[0]) for filename in model_id_checkpoints]
model_dir = os.path.join(os.getcwd(),"src","pytorch_nn","model")
model_file_name = "model_" + str(model_id) + "_epoch_" + str(max(epoch_numbers)) + ".pth"
state = torch.load(os.path.join(model_dir, model_file_name))
try:
    network.load_state_dict(state)
    pass
except RuntimeError:
    try:
        from modeling.sync_batchnorm.replicate import patch_replication_callback
        from modeling.deeplab import *
        network = DeepLab(num_classes=2)
        network.load_state_dict(state)
        pass
    except:
        raise NotImplementedError('ERROR: Model class from loaded checkpoint is not implemented yet:\n\t{}'.format(os.path.join(model_dir, model_file_name)))

epoch_offset = max(epoch_numbers)
print("loaded previous checkpoint ( {}): {}".format(epoch_offset, model_file_name))

# create specific output folder if not already created
output_folder = os.path.join(output_folder,"m"+str(model_id))
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

network.to(device)
network.eval()


iou_sum = 0
num_steps = 0
for step, (imgs, label_imgs, file_name) in enumerate(tqdm(val_loader)):
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            # get to device
            imgs = Variable(imgs).to(device) # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = label_imgs.squeeze(1)
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device) # (shape: (batch_size, img_h, img_w))

            # get output
            outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))
            outputs = outputs.squeeze(1)

            # get to cpu
            imgs = imgs.cpu()
            label_imgs = label_imgs.cpu()
            outputs = outputs.cpu()
            outputs = torch.argmax(outputs, dim=1)

            for idx in range(imgs.shape[0]):
                image = imgs[idx].cpu().detach().numpy().transpose((1,2,0))
                target = label_imgs[idx].cpu().detach().numpy().transpose((0,1))
                output = outputs[idx].cpu().detach().numpy().transpose((0,1))

                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                images = np.hstack((image_gray*255, target*255, output*255))
                cv2.imwrite(os.path.join(output_folder, file_name[0]),images)

            iou_sum += func.intersectionOverUnion(outputs,label_imgs)
            num_steps = step
print('total for IoU = {}'.format(iou_sum/num_steps))
