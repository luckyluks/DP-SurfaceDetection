import numpy as np
import cv2
from skimage import data, filters
import Functions as func
import alphashape as ash
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import Functions as f

img = cv2.imread('frame93.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)


threshPoints = np.where(thresh == 255)
points = []
for x in range(len(threshPoints[0])):
    points.append((threshPoints[0][x], threshPoints[1][x]))

alpha_shape = ash.alphashape(points, alpha=0.8)
contours = []
for polygon in alpha_shape:
    x, y = polygon.exterior.coords.xy
    current = []
    for i in range(len(x)):
        current.append([[int(y[i]), int(x[i])]])
    contours.append(np.array(current))
# Create hull array for convex hull points
hull = []

# Calculate points for each contour
for i in range(len(contours)):
    # Creating convex hull object for each contour
    hull.append(cv2.convexHull(contours[i], False))

# Create an empty black image
drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

# Draw contours and hull points
for i in range(len(contours)):
    color = (255, 255, 255)
    # Draw ith convex hull object and fill with white
    cv2.drawContours(drawing, hull, i, color, -1, 8)

drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
ref, thresh = cv2.threshold(drawing, 50, 255, cv2.THRESH_BINARY)







cv2.imshow("Alpha Hull", thresh)
cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
