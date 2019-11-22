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
hullframe, hull = func.CHI(dframe, 4, 50)
noArtframe = func.ArtFilt(hullframe, 300)


template = np.zeros(np.shape(dframe))

for object in hull:
    xlist = []
    ylist = []
    for point in object:
        xlist.append(point[0][0])
        ylist.append(point[0][1])

    x = min(xlist)
    y = min(ylist)
    w = max(xlist) - x
    h = max(ylist) - y
    rect = (x, y, w, h)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    if w != 0 and h != 0:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        frame = img * mask2[:, :, np.newaxis]
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        th, dframe = cv2.threshold(grayFrame, 1, 255, cv2.THRESH_BINARY)

        template = np.add(template, dframe)



col1 = np.hstack((np.true_divide(gframe,255), template))
col2 = np.hstack((hullframe, noArtframe))

col1 = np.vstack((col1, col2))
col1 = cv2.resize(col1, (960,540))

cv2.imshow('Frame', col1)
cv2.waitKey(0)