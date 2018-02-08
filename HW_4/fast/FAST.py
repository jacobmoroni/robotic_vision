import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('star.png',0)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create(threshold=25)
# fast = cv2.FAST_FEATURE_DETECTOR_TYPE_7_12

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, None,color=(255,0,0))
# cv2.imshow('image',img2)
# waitKey(0)
print"Threshold: ", fast.getThreshold()
print"nonmaxSuppression: ", fast.getNonmaxSuppression()
print"neighborhood: ", fast.getType()
print"Total Keypoints with nonmaxSuppression: ", len(kp)

cv2.imwrite('fast_true.png',img2)

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)

print "Total Keypoints without nonmaxSuppression: ", len(kp)

img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

cv2.imwrite('fast_false.png',img3)

while(1):
    cv2.imshow('Original',img)
    cv2.imshow('With Non-max Supression',img2)
    cv2.imshow('Without Non-max Supression',img3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
