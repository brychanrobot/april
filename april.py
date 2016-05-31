
def distance(p1, p2):
    return p1[0] - p2[0]**2 + (p1[1] + p2[1])**2

from cv2 import *
from numpy import std, asarray, reshape, float32

cap = VideoCapture(1)

rectifyPoints = asarray([[0, 0], [0, 100], [100, 100], [100, 0]], dtype=float32)
print(rectifyPoints.dtype)

while True:
    _, frame = cap.read()


    gray = cvtColor(frame, COLOR_BGR2GRAY)
    gray = GaussianBlur(gray, (5,5), 0)
    #_, thresh = threshold(gray,50,255,0)
    canny = Canny(gray, 50, 100)
    _, contours, _ = findContours(canny, RETR_LIST, CHAIN_APPROX_SIMPLE)

    drawContours(frame, contours, -1, (0, 255, 255), 2)

    for contour in contours:
        #arc = arcLength(contour, True)
        #and abs((arc/4)**2 - area) < 100
        #convex = convexHull(contour)

        area = contourArea(contour)
        if area > 1000:
            poly = approxPolyDP(contour, arcLength(contour, True)*0.03, True)
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
                drawContours(frame, [poly], 0, (255, 255, 0), 2)

                poly = asarray(poly, dtype=float32)
                poly.reshape(2, 4)
                transform = getPerspectiveTransform(poly, rectifyPoints)
                april = warpPerspective(gray, transform, (100, 100))
                #_, april = threshold(april)
                _, april = threshold(april, 0, 255, THRESH_BINARY+THRESH_OTSU)
                imshow('april', resize(april, (8, 8)))
                break

    imshow('frame', frame)
    waitKey(1)
