
def distance(p1, p2):
    return p1[0] - p2[0]**2 + (p1[1] + p2[1])**2

def hasBlackOutline(tag):
    numWhite = sum(tag[0, :]) + sum(tag[-1, :]) + sum(tag[:, 0]) + sum(tag[:,-1])
    return numWhite < 4

from cv2 import *
import numpy as np

seenTags = 0
tagMap = {}

cap = VideoCapture(1)

cap.set(CAP_PROP_FRAME_WIDTH,1920);
cap.set(CAP_PROP_FRAME_HEIGHT,1080);

rectifyPoints = np.asarray([[0, 0], [0, 100], [100, 100], [100, 0]], dtype=np.float32)

while True:
    _, frame = cap.read()


    gray = cvtColor(frame, COLOR_BGR2GRAY)
    gray = GaussianBlur(gray, (3,3), 0)
    #_, thresh = threshold(gray,50,255,0)
    canny = Canny(gray, 50, 250)
    _, contours, hierarchy = findContours(canny, RETR_TREE, CHAIN_APPROX_SIMPLE)

    #drawContours(frame, contours, -1, (0, 255, 255), 2)
    acceptedContours = []

    for c, contour in enumerate(contours):
        #arc = arcLength(contour, True)
        #and abs((arc/4)**2 - area) < 100
        #convex = convexHull(contour)
        if hierarchy[0][c][3] in acceptedContours:
            continue

        area = contourArea(contour)
        if area > 700:
            poly = approxPolyDP(contour, arcLength(contour, True)*0.1, True)
            if len(poly) == 4 and isContourConvex(poly):
                #print(poly[0])
                """
                lengths = [
                    distance(poly[0][0], poly[1][0]),
                    distance(poly[1][0], poly[2][0]),
                    distance(poly[2][0], poly[3][0]),
                    distance(poly[3][0], poly[0][0])
                ]
                """
                #lmax = amax(lengths)
                #lmin = amin(lengths)
                #affinity = (lengths[0] - lengths[2]) ** 2 + (lengths[1] - lengths[3]) ** 2
                #print(affinity)
                #if affinity < 40000000000:

                ndpoly = np.asarray(poly, dtype=np.float32)
                ndpoly.reshape(2, 4)
                transform = getPerspectiveTransform(ndpoly, rectifyPoints)
                april = warpPerspective(gray, transform, (100, 100))
                #_, april = threshold(april)
                _, april = threshold(april, 0, 255, THRESH_BINARY+THRESH_OTSU)
                april = resize(april, (8,8))

                if hasBlackOutline(april):
                    drawContours(frame, [poly], 0, (255, 255, 0), 2)

                    tagId = 0

                    for i, value in enumerate(np.nditer(april[1:7, 1:7])):
                        tagId = (tagId << i) | (1 if value > 230 else 0)

                    if tagId not in tagMap:
                        tagMap[tagId] = seenTags
                        seenTags += 1

                    acceptedContours.append(c)

                    centroid = np.ravel(np.sum(ndpoly, axis=0)/4)
                    putText(frame, str(tagMap[tagId]), tuple(centroid), FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255))

                    imshow('april', april)
                #break

    imshow('frame', frame)
    waitKey(1)
