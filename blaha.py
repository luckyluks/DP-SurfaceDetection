import numpy as np
import cv2

img = cv2.imread('frame93.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)
length = len(contours)

def dist_calc(cont1,cont2, min_dist):

    NOP1, NOP2 = cont1.shape[0], cont2.shape[0]

    for i in range(NOP1):
        for j in range(NOP2):

            dist = np.linalg.norm(cont1[i]-cont2[j])

            if abs(dist) < min_dist:
                return True
            elif i == NOP1-1 and j == NOP2-1:
                return False

# Pre-define a zeros-vector which denotes the "parent-hull" of each of the hulls to be connected with
contlist = np.zeros((length, 1))

# Loop through every contour and check the distance from it to every other contour. If it is close enough, put the same parent index on them, which is the smallest index that currently has been close to that constallation before
for n, cont1 in enumerate(contours[:length-1]):
    x = n
    for m, cont2 in enumerate(contours[n+1:]):
        x += 1
        dist = dist_calc(cont1, cont2, 50)
        if dist == True:
            lowestIndex = min(contlist[x],contlist[n])
            contlist[x] = contlist[n] = lowestIndex
        else:
            contlist[x] = n+1



maximum = int(contlist.max())+1
hullList = []
parents = np.unique(contlist)
print(contlist)
for i in range(maximum):
    pos = np.where(contlist == i)[0]
    if pos.size != 0:
        for j in pos:
            cont = np.vstack(contours[j])
            hull = cv2.convexHull(cont)
            hullList.append(hull)


"""
for i in range(len(parents)-1):
    if i < len(parents)-1:
        for j in range(int(parents[i]), int(parents[i+1])):
            cont = np.vstack(contours[j])
            hull = cv2.convexHull(cont)
            hullList.append(hull)

    else:
        for j in range(int(parents[i]), len(contlist)-1):
            cont = np.vstack(contours[j])
            hull = cv2.convexHull(cont)
            hullList.append(hull)

unified = []
maximum = int(contlist.max())+1
for i in range(maximum):
    pos = np.where(contlist==i)[0]
    if pos.size != 0:
        cont = np.vstack(contours[i] for i in pos)
        hull = cv2.convexHull(cont)
        unified.append(hull)
        
"""

cv2.drawContours(img,hullList,-1,(0,255,0),2)
cv2.drawContours(thresh,hullList,-1,255,-1)



cv2.imshow("Outline", img)
cv2.waitKey(0)
cv2.destroyAllWindows()





