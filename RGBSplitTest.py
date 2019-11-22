import numpy as np
import cv2
import Functions as func

# Open Video
cap = cv2.VideoCapture("object_detection_2.mp4")

# Select a image that represents the empty background
medianFrame = cv2.imread('bgapp.jpg')

# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Convert background to grayscale
bgR, bgG, bgB = func.ChannelSplit(medianFrame)

# Loop over all frames
ret = True
while(True):

  # Read frame
  ret, frame = cap.read()

  template = np.zeros(np.shape(bgR)[1])

  if ret:
    # Convert current frame to r,g and b
    fR, fG, fB = func.ChannelSplit(frame)
    # Calculate absolute difference of current frame and
    # the median frame
    dframeR = cv2.absdiff(fR, bgR)
    dframeG = cv2.absdiff(fG, bgG)
    dframeB = cv2.absdiff(fB, bgB)
    # Treshold to binarize
    th, dframeR = cv2.threshold(dframeR, 40, 255, cv2.THRESH_BINARY)
    th, dframeG = cv2.threshold(dframeR, 40, 255, cv2.THRESH_BINARY)
    th, dframeB = cv2.threshold(dframeR, 40, 255, cv2.THRESH_BINARY)

    dframeR = cv2.cvtColor(dframeR, cv2.COLOR_BGR2GRAY)
    dframeG = cv2.cvtColor(dframeG, cv2.COLOR_BGR2GRAY)
    dframeB = cv2.cvtColor(dframeB, cv2.COLOR_BGR2GRAY)

    template = np.add(template, dframeR)
    template = np.add(template, dframeG)
    template = np.add(template, dframeB)



    dframe = cv2.threshold(template, 1, 255, cv2.THRESH_BINARY)
    """
    # Do 2 passes to create a filled in convex hull of all moving objects
    hullframe, hull = func.CHI(dframe,4,50)
    # Remove small artifacts created by the background subtraction
    noArtframe = func.ArtFilt(hullframe, 300)

    #GrabCutFrame = func.GrabCut(hull, frame, dframe)

    col1 = np.hstack((np.true_divide(fR,255), dframe))
    col2 = np.hstack((hullframe, noArtframe))

    video = np.vstack((col1, col2))

    video = cv2.resize(video, (960, 540))
    """
    cv2.imshow('Video', template)
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
