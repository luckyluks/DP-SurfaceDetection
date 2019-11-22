import cv2
import numpy as np
import matplotlib.pyplot as plt
import Functions as func

img = cv2.imread('cframe93.jpg')
medianFrame = cv2.imread('bgapp.jpg')

grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
gframe = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dframe = cv2.absdiff(gframe, grayMedianFrame)
th, dframe = cv2.threshold(dframe, 40, 255, cv2.THRESH_BINARY)
hullframe, hull = func.CHI(dframe, 2, 50)
noArtframe = func.ArtFilt(hullframe, 300)


newHull = func.HullCombine(hull, 60)





"""
col1 = np.hstack((np.true_divide(gframe,255), dframe))
col2 = np.hstack((hullframe, noArtframe))

col1 = np.vstack((col1, col2))
col1 = cv2.resize(col1, (960,540))
"""
cv2.imshow('Frame', thresh)
cv2.waitKey(0)
