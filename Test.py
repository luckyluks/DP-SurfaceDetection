import numpy as np
import cv2
import Functions as func

img = cv2.imread('frame372.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

filt_img = func.ArtFilt(thresh, 50)

cv2.imshow("Filtered", filt_img)
cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()