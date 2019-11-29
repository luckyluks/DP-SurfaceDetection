import numpy as np
import cv2
from skimage import data, filters
import Functions as func

# Open Video
cap = cv2.VideoCapture("6.mp4")
bg = cv2.VideoCapture('Background.mp4')

# Method select: CH (Convex Hull), GC (GrabCut), rgbGC (Color GrabCut)
selection = 'rgbGC'

# Randomly select 25 frames
frameIds = bg.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = bg.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
if selection == 'rgbGC':
    rMedian, gMedian, bMedian = func.ChannelSplit(medianFrame)
else:
    grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# Loop over all frames
ret = True
while True:

    # Read frame
    ret, frame = cap.read()

    if ret:
        # Convert current frame to grayscale
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if selection == 'rgbGC':
          dframe = func.RGBConvexHull(frame, rMedian, gMedian, bMedian, 60, 60, 80)
        else:
          # Calculate absolute difference of current frame and
          # the median frame
          dframe = cv2.absdiff(gframe, grayMedianFrame)
          # Treshold to binarize
          th, dframe = cv2.threshold(dframe, 40, 255, cv2.THRESH_BINARY)


        # Do 2 passes to create a filled in convex hull of all moving objects
        hullframe, hull = func.CHI(dframe, 2, 50)
        # Remove small artifacts created by the background subtraction
        noArtframe = func.ArtFilt(hullframe, 100)

        if selection == 'GC' or selection == 'rgbGC':
            GrabCutFrame = func.GrabCut(hull, frame, dframe)

            col1 = np.hstack((np.true_divide(gframe,255), dframe))
            col2 = np.hstack((hullframe, GrabCutFrame))
        else:
            col1 = np.hstack((np.true_divide(gframe, 255), dframe))
            col2 = np.hstack((hullframe, noArtframe))

        video = np.vstack((col1, col2))

        video = cv2.resize(video, (960, 540))
        cv2.imshow('Video', video)
        cv2.waitKey(20)

    else:
        print('no more frames ... replaying video')
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Release video object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()

