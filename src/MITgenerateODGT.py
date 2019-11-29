import os

width = 640
height = 480

#Count frames
frameDict = {}
for subdir, dirs, files in os.walk('data'):
    if not len(dirs) == 0:
        if dirs[0].endswith('true'):
            for iDir in range(len(dirs)):
                curDir = os.path.join(subdir,dirs[iDir])
                nFrames = 0
                for subdirB, dirsB, filesB in os.walk(curDir):
                    for file in filesB:
                        filePath = os.path.join(subdir, file)
                        filename, file_extension = os.path.splitext(filePath)
                        if file_extension == '.jpg':
                            nFrames +=1
                    frameDict[curDir] = nFrames

#Generate ODGT file
with open("data/training.odgt", "w") as train_file:
    for ikey, ivalue in frameDict.items():
        name = ikey[16:-5]
        trueName = ikey[16:]
        for iLine in range(ivalue):
            tline1 = '{"fpath_img": "Frames/'+name+'/frame'+str(iLine+1)+'.jpg", "fpath_segm"'
            tline2 = ': "trueFrames/'+trueName+'/frame'+str(iLine+1)+'.jpg", "width": '+str(width)+', "height": '+str(height)+'}\n'
            tline = tline1+tline2
            train_file.write(tline)



# with open("data/validation.odgt", "w") as val_file:
#     val_file.write(vline)