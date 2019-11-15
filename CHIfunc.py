def CHI(thresh, passes):
    import numpy as np
    import cv2

    for i in range(1, passes):
        # Find contours of binary image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
            # Draw ith convex hull object
            cv2.drawContours(drawing, hull, i, color, -1, 8)

        drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
        ref, thresh = cv2.threshold(drawing, 50, 255, cv2.THRESH_BINARY)

    return thresh