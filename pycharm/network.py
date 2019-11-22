import numpy as np
import cv2
import Functions as func
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import ImageFolder
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import os

device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
torch.cuda.empty_cache()

gooNet = models.inception_v3(pretrained=True)


print('==================================================')
print('Model feature layers:')
print('==================================================')
for i in range(len(gooNet.fc)):
    print(gooNet.fc[i])

# print('==================================================')
# print('Model classifier layers:')
# print('==================================================')
# for i in range(len(gooNet.classifier)):
#     print(gooNet.classifier[i])

