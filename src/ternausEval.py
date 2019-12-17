import cv2
# from pylab import *
import torch
import numpy as np
from torch import nn
from ternausnet.unet_models import unet11
from pathlib import Path
from torch.nn import functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
import os
import Functions as func

#create network output folder
os.makedirs('data/outputUnetPT', exist_ok=True)

#pretrained true or 'carvana'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model():
    model = unet11(pretrained='carvana')   #or 'carvana'
    model.eval()
    return model.to(device)

def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of the car
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img


def load_image(path, pad=True):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)

    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not pad:
        return img

    height, width, _ = img.shape

    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def crop_image(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]



#if no shuffle (big dataset)
# np.random.seed(1337)
# order = np.random.randint(4662,6660,size=500)
# if between new order FIX

#else use all images
order = np.linspace(4502,5405,904).astype('int')



iousum = 0
for imn in range(500):

    if(imn % 50 == 0):
        print(imn)

    index = order[imn]
    img, pads = load_image('data/Frames/frame'+str(index)+'.jpg', pad=True)
    groundtruth, pads = load_image('data/trueFrames/frame'+str(index)+'.jpg', pad=True)

    with torch.no_grad():
        input_img = torch.unsqueeze(img_transform(img).to(device), dim=0)
        mask = torch.sigmoid(model(input_img))

    mask_array = mask.data[0].cpu().numpy()[0]
    mask_array = crop_image(mask_array, pads)
    prediction = (mask_array > 0.5).astype(np.uint8)

    c2 = np.hstack((mask_array, prediction))
    comb = np.round(c2 * 255)
    comb = comb.astype(np.uint8)


    cv2.imwrite('data/outputUnetPT/frame'+str(index)+'.jpg', comb)

    groundtruth = groundtruth[:,:,0]
    groundtruth = (groundtruth/255).astype(np.uint8)

    iousum += func.intersectionOverUnion(prediction,groundtruth)

print('total IoU = {}'.format(iousum/500))