import cv2
import numpy as np

video = cv2.VideoCapture('video.mp4')
MOG = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

while True:
    _, frame = video.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    MOGframe = MOG.apply(grayFrame)

    cv2.imshow("MOG", MOGframe)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
