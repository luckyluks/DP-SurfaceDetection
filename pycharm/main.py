import numpy as np
import cv2
import Functions as func


from skimage import data, filters

nFramesUsed = 20
vSpeed = 50


#catch bg from bg video
bgvid = cv2.VideoCapture("venv/include/bgC.mp4")

#input video
cap = cv2.VideoCapture("venv/include/train1Color.mp4")

# Randomly select 25 frames
# frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)


frameIds = np.linspace(1,nFramesUsed,nFramesUsed)

# Store selected frames in an array
# frames = []
# print("used frames for median:", frameIds)
# for fid in frameIds:
#     bgvid.set(cv2.CAP_PROP_POS_FRAMES, fid)
#     ret, frame = bgvid.read()
#     frames.append(frame)


# medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

# for gimp
# cv2.imwrite('bg_median.jpg', medianFrame)

medianFrame = cv2.imread("bgapp.jpg")

# Display median frame
cv2.imshow('frame', medianFrame)
cv2.waitKey(0)

# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# Loop over all frames
stop = False
while not stop:

  # Read frame
  ret, frame = cap.read()

  if ret:
    # Convert current frame to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate absolute difference of current frame and
    # the median frame
    dframe = cv2.absdiff(frame, grayMedianFrame)
    # Treshold to binarize
    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)

    # dframe = func.ConvexHull(dframe,50)
    dframe = func.CHI(dframe,2,50)
    dframe = func.ArtFilt(dframe,50)



    # Display image, original image
    dframe = np.hstack((np.true_divide(frame,255), dframe))

    cv2.imshow('dframe', dframe)
    cv2.waitKey(vSpeed)
  else:
    print('no more frames ... replaying with p, quit with q')
    if cv2.waitKey(0) == ord('p'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    elif cv2.waitKey(0) == ord('q'):
        stop=True


# Release video object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()