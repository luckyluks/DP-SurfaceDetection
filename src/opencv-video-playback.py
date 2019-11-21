import numpy as np
import cv2

cap = cv2.VideoCapture('recordings/object_detection_depth_2.mp4')
print(cap.isOpened())

# Loop over all frames
ret = True
while(True):

  # Read frame
  ret, frame = cap.read()
#   print(ret)

  if ret:
    # print("ret")
    # Convert current frame to grayscale
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate absolute difference of current frame and 
    # the median frame
    # dframe = cv2.absdiff(frame, grayMedianFrame)
    # # Treshold to binarize
    # th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
    # Display image
    frame_colormap = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    cv2.imshow('frame', frame_colormap)
    cv2.waitKey(20)
  else:
    print('no more frames ... replaying video')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
    break


# Release video object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()