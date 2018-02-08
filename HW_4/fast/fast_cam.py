import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create(threshold=25)

def nothing(x):
    pass
cv2.namedWindow('FAST')
switch0 = 'greyscale (on/off)'
cv2.createTrackbar(switch0,'FAST',0,1,nothing)
switch = 'Nonmax Supression (on/off)'
cv2.createTrackbar(switch, 'FAST',1,1,nothing)
cv2.createTrackbar('threshold','FAST',50,255,nothing)
# cv2.createTrackbar('neighborhood','FAST',2,2,nothing)


while True:
    ret, frame = cap.read()

    thresh = cv2.getTrackbarPos('threshold','FAST')
    # nbhood = cv2.getTrackbarPos('neighborhood','FAST')

    # cv2.getTrackbarPos('neighborhood','FAST')
    s = cv2.getTrackbarPos(switch,'FAST')
    s0 = cv2.getTrackbarPos(switch0,'FAST')

    if s0 ==1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# find and draw the keypoints
    kp = fast.detect(frame,None)
    img2 = cv2.drawKeypoints(frame, kp, None,color=(255,0,0))
    cv2.imshow('FAST',img2)





    fast.setNonmaxSuppression(s)
    fast.setThreshold(thresh)
    # fast.setType(nbhood)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
