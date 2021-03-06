import numpy as np
import cv2
import Functions as func
import os

# This script goes through a folder structure, looking for background videos starting with 'bg' and then applying the
# full background subtraction on each frame in the rest of the videos in the same folder as the background video,
# assumed recorded in the same setting and produce and saves the output to some certain output folders

#decide color thresholds
rThresh = 60
gThresh = 60
bThresh = 60

#declare input data path
dataPath = 'venv/include/'
folders = []
nrOfTotal = 0
for subdirs, dirs, files in os.walk(dataPath):
    nrOfTotal += len(files)
    if(len(subdirs)>len(dataPath)):
        info = {}
        info["name"] = subdirs
        info["bgPath"] = [subdirs+'/'+bg for bg in files if bg.startswith('bg')][0]
        info["videos"] = [subdirs+'/'+vid for vid in files if not vid.startswith('bg')]
        info["new"] = len([True for ncheck in files if ncheck.startswith('new')])>0
        folders.append(info)
nrOfTotal = (nrOfTotal-len(folders))*180

# Create output folders
os.makedirs('data/Frames/', exist_ok=True)
os.makedirs('data/trueFrames/', exist_ok=True)

foldcount = 1
totalCounter = 1
for fold in folders:

    # BG video
    bgvid = cv2.VideoCapture(fold.get('bgPath'))

    # Randomly select 25 frames
    frameIds = bgvid.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        bgvid.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = bgvid.read()
        frames.append(frame)

    # Convert to 8bit int
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

    # Convert background to grayscale
    grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

    #Split frame into color channels
    rMedian, gMedian, bMedian = func.ChannelSplit(medianFrame)

    vidcount = 1
    for video in fold.get('videos'):

        # Input video
        videoPath = video
        cap = cv2.VideoCapture(videoPath)

        #reset frame counter
        frameCounter = 1

        #get nr of frames
        nrOfFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        stop = False
        while not stop:

            # Read frame
            ret, frame = cap.read()

            if not fold.get('new') and ((frameCounter-1) % 5 == 0): #only for old videos recorded at 30 fps, remove otherwise

                # Save frame
                cv2.imwrite('data/Frames/frame' + str(totalCounter) + '.jpg', frame)

                #FILTERING
                dframe = func.RGBConvexHull(frame, rMedian, gMedian, bMedian, rThresh, gThresh, bThresh)
                sureFrame = func.RGBConvexHull(frame, rMedian, gMedian, bMedian, 100, 100, 100)
                surebgFrame = func.RGBConvexHull(frame, rMedian, gMedian, bMedian, 30, 30, 30)

                # Do 2 passes to create a filled in convex hull of all moving objects
                hullframe, hull = func.CHI(dframe, 2, 50)
                # Remove small artifacts created by the background subtraction
                noArtframe = func.ArtFilt(hullframe, 100)


                noArtframe = noArtframe.astype(np.uint8)
                hullframe, hull = func.CHI(noArtframe, 2, 50)
                filteredFrame = func.GrabCutPixel(hullframe, frame, dframe, sureFrame, surebgFrame)

                # save truth (filtered frame)
                U8frameg = np.uint8(filteredFrame)
                U8frame = cv2.cvtColor(U8frameg, cv2.COLOR_GRAY2BGR)
                cv2.imwrite('data/trueFrames/frame' + str(totalCounter) + '.jpg', U8frame)

                # print progress
                os.system('clear')
                print('folder {} of {} | file {} of {} | frame {} of {} filename: {} | frame {}'.format(foldcount,len(folders),vidcount,len(fold.get('videos')),totalCounter,nrOfTotal,videoPath,frameCounter),flush=True)

                #increase filename counter
                totalCounter +=1
            elif fold.get('new'):
                # Save frame
                cv2.imwrite('data/Frames/frame' + str(totalCounter) + '.jpg', frame)

                # FILTERING
                dframe = func.RGBConvexHull(frame, rMedian, gMedian, bMedian, rThresh, gThresh, bThresh)
                sureFrame = func.RGBConvexHull(frame, rMedian, gMedian, bMedian, 100, 100, 100)
                surebgFrame = func.RGBConvexHull(frame, rMedian, gMedian, bMedian, 30, 30, 30)

                # Do 2 passes to create a filled in convex hull of all moving objects
                hullframe, hull = func.CHI(dframe, 2, 50)
                # Remove small artifacts created by the background subtraction
                noArtframe = func.ArtFilt(hullframe, 100)

                noArtframe = noArtframe.astype(np.uint8)
                hullframe, hull = func.CHI(noArtframe, 2, 50)
                filteredFrame = func.GrabCutPixel(hullframe, frame, dframe, sureFrame, surebgFrame)

                # save truth (filtered frame)
                U8frameg = np.uint8(filteredFrame)
                U8frame = cv2.cvtColor(U8frameg, cv2.COLOR_GRAY2BGR)
                cv2.imwrite('data/trueFrames/frame' + str(totalCounter) + '.jpg', U8frame)

                # print progress
                os.system('clear')
                print('folder {} of {} | file {} of {} | frame {} of {} filename: {} | frame {}'.format(foldcount, len(folders),
                                                                                                        vidcount, len(fold.get('videos')),
                                                                                                        totalCounter, nrOfTotal, videoPath, frameCounter), flush=True)
                # increase filename counter
                totalCounter += 1
            else:
                nothing = []

            frameCounter +=1
            if frameCounter > nrOfFrames:
                stop = True
                print('no more frames, recording done')

        # Release video object
        cap.release()
        vidcount +=1

    foldcount +=1



