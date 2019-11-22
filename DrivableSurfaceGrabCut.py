import numpy as np
import cv2
from skimage import data, filters
import Functions as func

# Open Video
cap = cv2.VideoCapture("object_detection_2.mp4")

# Select a image that represents the empty background
medianFrame = cv2.imread('bgapp.jpg')

# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# Loop over all frames
ret = True
while(True):

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
    hullframe, hull = func.CHI(dframe,4,50)
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

cap = cv2.imread('map.jpg')
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

x = 50
y = 50
w = 250
h = 300

rect = (x, y, w, h)

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

frame = img*mask2[:, :, np.newaxis]

# Convert background to grayscale
grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Treshold to binarize
th, dframe = cv2.threshold(grayFrame, 1, 255, cv2.THRESH_BINARY)

cv2.imshow('Mask', dframe)
cv2.waitKey(0)
