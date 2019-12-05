import os
import numpy as np

width = 640
height = 480

#Count frames
for subdir, dirs, files in os.walk('data/trueFrames'):
    nrOfFrames = len(files)

#Define split train/validation
trainSplit = np.int(np.round(0.99*nrOfFrames))

#Randomize order
# order = np.random.permutation(nrOfFrames)+1
order = np.linspace(1,nrOfFrames,nrOfFrames, dtype=int)

#Generate ODGT files
with open("data/training.odgt", "w") as train_file:
    for iLine in range(trainSplit):
        index = order[iLine]
        tline1 = '{"fpath_img": "Frames/frame'+str(index)+'.jpg", "fpath_segm"'
        tline2 = ': "trueFrames/frame'+str(index)+'.jpg", "width": '+str(width)+', "height": '+str(height)+'}\n'
        tline = tline1+tline2
        train_file.write(tline)

with open("data/validation.odgt", "w") as val_file:
    for iLine in range(nrOfFrames-trainSplit):
        index = order[iLine+trainSplit]
        vline1 = '{"fpath_img": "Frames/frame' + str(index) + '.jpg", "fpath_segm"'
        vline2 = ': "trueFrames/frame' + str(index) + '.jpg", "width": ' + str(width) + ', "height": ' + str(
            height) + '}\n'
        vline = vline1 + vline2
        val_file.write(vline)