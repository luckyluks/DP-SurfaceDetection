import numpy as np
import cv2
import Functions as func
import os

outpf = 'data/combtwo'
inleft = 'data/trueFrames'
inright = 'data/trueFramesNEW'

# Create output folders
os.makedirs(outpf, exist_ok=True)


np.random.seed(1337)
order = np.random.randint(1,7564,size=500)

for i in range(500):
    index = order[i]
    real = cv2.imread(inleft+'/frame'+str(index)+'.jpg')
    true = cv2.imread(inright+'/frame'+str(index)+'.jpg')

    comb = np.vstack((real,true))

    cv2.imwrite(outpf+'/frame'+str(index)+'.jpg', comb)


