#basic imports
import numpy as np
import cv2
import torch
import os
import yaml

#own imports
import Functions as func

#imports from MIT-sem-segm model
import MITnet.config
import MITnet.dataset
import MITnet.lib.nn
import MITnet.models
import MITnet.utils
import MITnet.train
import MITnet.eval

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
cfg = MITnet.config.cfg
#this model has good performance
cfg.merge_from_file("MITnet/config/ade20k-mobilenetv2dilated-c1_deepsup.yaml")

#check for use of cuda
device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
print("using:",device)
torch.cuda.empty_cache()


gpus = [0]
MITnet.train.main(cfg,gpus)

