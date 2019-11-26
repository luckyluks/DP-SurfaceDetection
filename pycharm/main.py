import numpy as np
import cv2
import Functions as func
import pyrealsense2 as rs
import os

from skimage import data, filters

# HOWTO: set exposure for
# p = rs.pipeline()
# prof = p.start()
# s = prof.get_device().query_sensor'data/trueFrames/'+fileName+'-true/frame'+str(frameCounter)+'.jpg's()[1]
# s.set_option(rs.option.exposure, new_value)
# rs.option_range.default


# nFramesUsed = 20
vSpeed = 50
record = True

#BG video
bgvid = cv2.VideoCapture("venv/include/BGCvNOV22.mp4")

#Input video
fileName = 'o_nov22-1'
cap = cv2.VideoCapture('venv/include/'+fileName+'.mp4')

#Create output folders
os.makedirs('data/Frames/'+fileName, exist_ok=True)
os.makedirs('data/trueFrames/'+fileName+'-true', exist_ok=True)

#Get video data
frameWidth = int(cap.get(3))
frameHeight = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
nrOfFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(nrOfFrames)

#Init videowriter
# truthOut = cv2.VideoWriter('truth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frameWidth, frameHeight))

# Randomly select 25 frames
# frameIds = np.linspace(1,nFramesUsed,nFramesUsed)
frameIds = bgvid.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

# Store selected frames in an array
frames = []
print("used frames for median:", frameIds)
for fid in frameIds:
    bgvid.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = bgvid.read()
    frames.append(frame)

# for fid in frameIds:
#     cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
#     ret, frame = cap.read()
#     frames.append(frame)


medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

# for gimp (manual background)
# cv2.imwrite('bg_median.jpg', medianFrame)
# medianFrame = cv2.imread("bgapp.jpg")

# Display median frame
cv2.imshow('frame', medianFrame)
cv2.waitKey(0)

# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)


# Loop over all frames
if not record:
    stop = False
    i = 0
    while not stop:

      # Read frame
      ret, frame = cap.read()

      if ret:
        i+=1
        # Convert current frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate absolute difference of current frame and
        # the median frame
        rawdiff = cv2.absdiff(frame, grayMedianFrame)
        # Treshold to binarize
        th, binaryframe = cv2.threshold(rawdiff, 30, 255, cv2.THRESH_BINARY)

        # dframe = func.ConvexHull(dframe,50)
        filteredFrame = func.CHI(binaryframe,2,50)
        filteredFrame = func.ArtFilt(filteredFrame,50)

        # Display image, original image
        sframe1 = np.hstack((np.true_divide(frame,255), np.true_divide(rawdiff,255)))
        sframe2 = np.hstack((binaryframe, filteredFrame))

        allframes = np.vstack((sframe1,sframe2))

        cv2.imshow('allframes', allframes)
        cv2.waitKey(vSpeed)
      else:
        print('no more frames ... replaying with p, quit with q')
        if cv2.waitKey(0) == ord('p'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif cv2.waitKey(0) == ord('q'):
            stop=True


    # Release video object
    cap.release()
    # truthOut.release()

    # Destroy all windows
    cv2.destroyAllWindows()
    print(i)
else:
    frameCounter = 1
    stop = False
    while not stop:

        # Read frame
        ret, frame = cap.read()

        # Save frame
        cv2.imwrite('data/Frames/'+fileName+'/frame'+str(frameCounter)+'.jpg', frame)

        # Convert current frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate absolute difference of current frame and
        # the median frame
        rawdiff = cv2.absdiff(frame, grayMedianFrame)
        # Treshold to binarize
        th, binaryframe = cv2.threshold(rawdiff, 30, 255, cv2.THRESH_BINARY)

        # dframe = func.ConvexHull(dframe,50)
        filteredFrame = func.CHI(binaryframe, 2, 50)
        filteredFrame = func.ArtFilt(filteredFrame, 50)

        # save truth (filtered frame)
        U8frame = np.uint8(filteredFrame)
        U8frame = cv2.cvtColor(U8frame, cv2.COLOR_GRAY2BGR)
        # truthOut.write(U8frame)
        cv2.imwrite('data/trueFrames/'+fileName+'-true/frame'+str(frameCounter)+'.jpg', U8frame)

        # Display image, original image
        sframe1 = np.hstack((np.true_divide(frame, 255), np.true_divide(rawdiff, 255)))
        sframe2 = np.hstack((binaryframe, filteredFrame))

        allframes = np.vstack((sframe1, sframe2))

        cv2.imshow('allframes', allframes)
        cv2.waitKey(vSpeed)

        frameCounter += 1
        if frameCounter > nrOfFrames:
            stop = True
            print('no more frames, recording done')


    # Release video object
    cap.release()
    # truthOut.release()

    # Destroy all windows
    cv2.destroyAllWindows()
    print(frameCounter)