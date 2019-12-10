import os
import numpy as np

def get_names():

    width = 640
    height = 480

    #Count frames
    for subdir, dirs, files in os.walk('data/trueFrames'):
        nrOfFrames = len(files)

    #Define split train/validation
    trainSplit = np.int(np.round(0.7*nrOfFrames))

    #Randomize order
    # order = np.random.permutation(nrOfFrames)+1
    order = np.linspace(1,nrOfFrames,nrOfFrames, dtype=int)

    #Generate lists of filenames
    train_file_names = []
    val_file_names = []

    for iLine in range(trainSplit):
        index = order[iLine]
        tline1 = 'data/Frames/frame'+str(index)+'.jpg'
        # tline2 = 'data/trueFrames/frame'+str(index)+'.jpg'
        train_file_names.append(tline1)
        # train_file_names.append(tline2)

    for iLine in range(nrOfFrames-trainSplit):
        index = order[iLine+trainSplit]
        vline1 = 'data/Frames/frame' + str(index) + '.jpg'
        # vline2 = 'data/trueFrames/frame' + str(index) + '.jpg'
        val_file_names.append(vline1)
        # val_file_names.append(vline2)


    return train_file_names, val_file_names