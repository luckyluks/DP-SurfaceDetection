#basic imports
import numpy as np
import cv2
import torch
import os
import yaml
import caffe2


#own imports
import Functions as func

#imports from MIT-sem-segm model
import config
import dataset
import lib.nn
import models
import utils
import train

#deep learning imports
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import ImageFolder
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms

#load config
cfg = config.cfg
#this model has good performance
cfg.merge_from_file("config/ade20k-mobilenetv2dilated-c1_deepsup.yaml")

#check for use of cuda
device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
print("using:",device)
torch.cuda.empty_cache()


gpus = 0
train.main(cfg,gpus)

