import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('map.jpg')
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

x = 0
y = 0
w = 250
h = 300

rect = (x, y, w, h)

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

frame = img*mask2[:, :, np.newaxis]

# Convert background to grayscale
grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Treshold to binarize
th, dframe = cv2.threshold(grayFrame, 1, 255, cv2.THRESH_BINARY)

cv2.imshow('Mask', dframe)
cv2.waitKey(0)
