import numpy as np
import cv2
import os

# This script was used to combine certain images together for comparison reasons

outputFolder = 'data/combtwo'

first = 'data/trueFramesOLD'
second = 'data/trueFrames'
third = 'data/Frames'

# Create output folders
os.makedirs(outputFolder, exist_ok=True)

for i in range(7564):

    index = i+1
    a = cv2.imread(first+'/frame'+str(index)+'.jpg')
    b = cv2.imread(second+'/frame'+str(index)+'.jpg')
    c = cv2.imread(third+'/frame'+str(index)+'.jpg')

    comb = np.vstack((a,b,c))
    print(i)
    cv2.imwrite(outputFolder+'/frame'+str(index)+'.jpg', comb)


