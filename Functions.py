def CHI(thresh, passes, cutoff):
    import numpy as np
    import cv2

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

    return thresh

def ArtFilt(img, min_size):
    import numpy as np
    import cv2

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
    import numpy as np
    import cv2

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
    import numpy as np

    NOP1, NOP2 = cont1.shape[0], cont2.shape[0]

    for i in range(NOP1):
        for j in range(NOP2):

            dist = np.linalg.norm(cont1[i]-cont2[j])

            if abs(dist) < min_dist:
                return True
            elif i == NOP1-1 and j == NOP2-1:
                return False