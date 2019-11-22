import numpy as np
import cv2
from skimage import data, filters
import Functions as func

# Open Video
cap = cv2.VideoCapture("2.mp4")
bg = cv2.VideoCapture('Background.mp4')

frameWidth = int(cap.get(3))
frameHeight = int(cap.get(4))

rawOut = cv2.VideoWriter('raw.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frameWidth, frameHeight))
truthOut = cv2.VideoWriter('truth.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frameWidth, frameHeight))

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

# Loop over all frames
ret = True
while True:

  # Read frame
  ret, frame = cap.read()

  if ret:
    # Convert current frame to grayscale
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate absolute difference of current frame and
    # the median frame
    dframe = cv2.absdiff(gframe, grayMedianFrame)
    # Treshold to binarize
    th, dframe = cv2.threshold(dframe, 40, 255, cv2.THRESH_BINARY)
    # Do 2 passes to create a filled in convex hull of all moving objects
    hullframe, hull = func.CHI(dframe, 3, 50)
    # Remove small artifacts created by the background subtraction
    noArtframe = func.ArtFilt(hullframe, 300)

    GrabCutFrame = func.GrabCut(hull, frame, dframe)

    rawOut.write(frame)
    truthOut.write(GrabCutFrame)

  else:
      break

rawOut.release()
truthOut.release()


