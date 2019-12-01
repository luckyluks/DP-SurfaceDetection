import numpy as np
import cv2
from skimage import data, filters
import platform

# Open Video
print(platform.system())
if platform.system()=="Windows":
  print("windows os detected")
  # cap = cv2.VideoCapture("C:\\Users\\lukas\\workspace\\DP-SurfaceDetection\\example_data\\video.mp4")
  cap = cv2.VideoCapture("C:\\Users\\lukas\\workspace\\DP-SurfaceDetection\\recordings\\old\\object_detection_2.mp4")
elif platform.system()=="Linux":
  print("no windows os detected")
  cap = cv2.VideoCapture("/home/zed/workspace/DP-SurfaceDetection/example_data/video.mp4")
else: 
  print("error: unsupported plattform")
  exit(0)

frames_count_initial = 2
frame_start_initial = 30

# Randomly select 25 frames
# frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=frames_count_initial)
frameIds = range(frame_start_initial, frame_start_initial+frames_count_initial)

# Store selected frames in an array
frames = []
print("used frames for median:", len(frameIds))
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

# Display median frame
cv2.imshow('frame', medianFrame)
cv2.waitKey(0)

# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# Loop over all frames
ret = True
frame_counter = 1
while(True):

  # Read frame
  ret, frame_org = cap.read()

  if ret:
    # Convert current frame to grayscale
    frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2GRAY)
    # Calculate absolute difference of current frame and 
    # the median frame
    dframe = cv2.absdiff(frame, grayMedianFrame)
    # Treshold to binarize
    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
    # Display image
    cv2.imshow('frame', dframe)
    cv2.waitKey(20)
  else:
    print('no more frames ... replaying video')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  

  if frame_counter % 50 ==0:
    frames.append(frame_org)
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    print("added frame to median. new frames size:", len(frames))

  frame_counter+=1

# Release video object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()