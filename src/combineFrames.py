import numpy as np
import cv2
import Functions as func
import os


# Create output folders
os.makedirs('data/combOut', exist_ok=True)

for i in range(7564):
    real = cv2.imread('data/Frames/frame'+str(i+1)+'.jpg')
    true = cv2.imread('data/trueFrames/frame'+str(i+1)+'.jpg')

    comb = np.hstack((real,true))
    cv2.imwrite('data/combOut/frame'+str(i+1)+'.jpg',comb)


