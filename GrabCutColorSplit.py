import numpy as np
import cv2
from skimage import data, filters
import Functions as func

# Open Video
cap = cv2.VideoCapture("object_detection_30s_20191126-160837.mp4")
bg = cv2.VideoCapture('bg_outsidelab_nov26.mp4')

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

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
rMedian, gMedian, bMedian = func.ChannelSplit(medianFrame)

# Loop over all frames
ret = True
while True:

  # Read frame
  ret, frame = cap.read()

  if ret:

    dframe = func.RGBConvexHull(frame, rMedian, gMedian, bMedian)
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Do 2 passes to create a filled in convex hull of all moving objects
    hullframe, hull = func.CHI(dframe, 2, 50)
    # Remove small artifacts created by the background subtraction
    noArtframe = func.ArtFilt(hullframe, 300)

    GrabCutFrame = func.GrabCut(hull, frame, dframe)

    col1 = np.hstack((np.true_divide(gframe,255), dframe))
    col2 = np.hstack((GrabCutFrame, noArtframe))

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
