import numpy as np
import cv2
from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point

# Decide which hulls are actually inside other hulls, and just keep those who are not
finalHull = list(range(1,len(hull))) # Enumerate all hulls in a list
n = 0
m = 0
# Loop over all individual hulls
for shape1 in hull:
    n += 1
    polygonList = []
    for node in shape1:
        for coord in node:
            polygonList.append((coord[0], coord[1]))

    # If the hull in question is 2D (i.e. not a line or a point)
    if len(polygonList) > 2:
        polygon = Polygon(np.asarray(polygonList))

        # Check all hulls if their points are inside the hull in question
        for shape2 in hull:
            pointList = []
            m += 1
            insidePointIndicator = True
            insideIndicator = True

            for point in shape2:
                for index in point:
                    insidePointIndicator = polygon.contains(Point(index[0],index[1]))

                    if insidePointIndicator == False:
                        insideIndicator = False

            # If all points are inside the hull, remove the second hull from the list (because it is redundant)
            if insideIndicator == True:
                if m in finalHull:
                    finalHull.remove(m)
    # If the hull is a line or point, remove it from the list
    else:
        if n in finalHull:
            finalHull.remove(n)

print(finalHull)

newHull = []
for i in finalHull:
    # Creating convex hull object for each contour
    newHull.append(hull[i])
