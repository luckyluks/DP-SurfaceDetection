import numpy as np
import cv2
import alphashape as ash

def CHI(thresh, passes, cutoff):
    for i in range(1, passes):
        # Find contours of binary image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        ref, thresh = cv2.threshold(drawing, cutoff, 255, cv2.THRESH_BINARY)

    return thresh, hull

def ArtFilt(img, min_size):
    # Find all your connected components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    # Result
    img2 = np.zeros((output.shape))
    # For every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2

def ConvexHull(thresh, min_dist):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)
    length = len(contours)

    # Pre-define a zeros-vector which denotes the "parent-hull" of each of the hulls to be connected with
    contlist = np.zeros((length, 1))

    # Loop through every contour and check the distance from it to every other contour. If it is close enough, put the same parent index on them, which is the smallest index that currently has been close to that constallation before
    for n, cont1 in enumerate(contours[:length - 2]):
        x = n
        for m, cont2 in enumerate(contours[n + 1:]):
            x += 1
            dist = dist_calc(cont1, cont2, min_dist)
            if dist == True:
                lowestIndex = min(contlist[x], contlist[n])
                contlist[x] = contlist[n] = lowestIndex
            else:
                contlist[x] = n + 1

    maximum = int(contlist.max()) + 1
    hullList = []
    #parents = np.unique(contlist)
    for i in range(maximum):
        pos = np.where(contlist == i)[0]
        if pos.size != 0:
            for j in pos:
                cont = np.vstack(contours[j])
                hull = cv2.convexHull(cont)
                hullList.append(hull)

    cv2.drawContours(thresh, hullList, -1, 255, -1)

    return thresh

def dist_calc(cont1,cont2, min_dist):
    NOP1, NOP2 = cont1.shape[0], cont2.shape[0]

    for i in range(NOP1):
        for j in range(NOP2):

            dist = np.linalg.norm(cont1[i]-cont2[j])

            if abs(dist) < min_dist:
                return True
            elif i == NOP1-1 and j == NOP2-1:
                return False

def GrabCut(hull, img, dframe):
    # Create empty matrix for adding the different detected objects
    template = np.zeros(np.shape(dframe))

    # For every object detected in the hull, create a bounding box around that hull and preform GrabCut on it
    # Then merge all these GrabCut results
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

    return template

def GrabCutPixel(hullframe, img, dframe, sureFrame, surebgFrame):
    # Create empty matrix for adding the different detected objects
    template = np.zeros(np.shape(dframe))

    # For every object detected in the hull, create a bounding box around that hull and preform GrabCut on it
    # Then merge all these GrabCut results
    mask = np.ones(img.shape[:2], np.uint8)*2
    mask[surebgFrame==0] = 0
    mask[hullframe==255] = 3
    mask[sureFrame==255] = 1

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    if (1 or 3) in mask:
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    else:
        mask2 = np.zeros(img.shape[:2], np.uint8)

    frame = img * mask2[:, :, np.newaxis]
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th, dframe = cv2.threshold(grayFrame, 1, 255, cv2.THRESH_BINARY)

    template = np.add(template, dframe)

    return template

def ChannelSplit(image):
    i = image.copy()
    # Separate the 3 color channels
    b = i[:, :, 0]

    g = i[:, :, 1]

    r = i[:, :, 2]

    return r, g, b

def RGBConvexHull(frame, rMedian, gMedian, bMedian, rT, gT, bT):

    rFrame, gFrame, bFrame = ChannelSplit(frame)
    rDiffFrame = cv2.absdiff(rFrame, rMedian)
    gDiffFrame = cv2.absdiff(gFrame, gMedian)
    bDiffFrame = cv2.absdiff(bFrame, bMedian)
    # Treshold to binarize
    _, rDiffFrame = cv2.threshold(rDiffFrame, rT, 255, cv2.THRESH_BINARY)
    _, gDiffFrame = cv2.threshold(gDiffFrame, gT, 255, cv2.THRESH_BINARY)
    _, bDiffFrame = cv2.threshold(bDiffFrame, bT, 255, cv2.THRESH_BINARY)

    dframe = np.add(np.add(np.asarray(rDiffFrame), np.asarray(gDiffFrame)), np.asarray(bDiffFrame))

    dframe[dframe > 0] = 255

    return dframe

def HullCombine(hull, minDist, dframe):
    c1 = -1
    objList = np.zeros(np.shape(hull)[0])
    for object1 in hull[:-1]:
        c1 += 1
        c2 = c1
        for object2 in hull[c1 + 1:]:
            merge = False
            c2 += 1
            for point1 in object1:
                for point2 in object2:

                    dist = np.linalg.norm(point1[0]-point2[0])

                    if dist < minDist:
                        merge = True
            if merge:
                objList[c2] = objList[c1]
            else:
                objList[c2] = c1 + 1

    finalHull = []
    for b in range(len(objList)):
        blist = []
        for j in range(len(objList)):
            if objList[j] == b:
                blist.append(hull[j][0])
        if blist:
            blist = np.array(blist)
            finalHull.append(blist)

    # Create an empty black image
    drawing = np.zeros((dframe.shape[0], dframe.shape[1], 3), np.uint8)

    # Draw contours and hull points
    for i in range(len(finalHull)):
        color = (255, 255, 255)
        # Draw ith convex hull object and fill with white
        cv2.drawContours(drawing, finalHull, i, color, -1, 8)

    drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    ref, thresh = cv2.threshold(drawing, 40, 255, cv2.THRESH_BINARY)

def AlphaHull(thresh, alpha, cutoff):
    threshPoints = np.where(thresh == 255)
    points = []
    for x in range(len(threshPoints[0])):
        points.append((threshPoints[0][x], threshPoints[1][x]))

    alpha_shape = ash.alphashape(points, alpha=alpha)
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
    ref, thresh = cv2.threshold(drawing, cutoff, 255, cv2.THRESH_BINARY)

    return thresh, hull


def intersectionOverUnion(imPred, imLabel):
    imPred = np.asarray(imPred).copy()
    imLabel = np.asarray(imLabel).copy()

    #go from 0/1 to 1/2 (depending on what are known as unlabeled)
    imPred += 1
    imLabel += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLabel > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLabel)
    (area_intersection, _) = np.histogram(
        intersection, bins=2, range=(1, 2))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=2, range=(1, 2))
    (area_lab, _) = np.histogram(imLabel, bins=2, range=(1, 2))
    area_union = area_pred + area_lab - area_intersection

    return area_intersection/area_union
