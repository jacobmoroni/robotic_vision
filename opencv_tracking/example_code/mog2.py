import numpy as np
import cv2
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history = 2,detectShadows = False)
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv2.erode(fgmask,(3,3),iterations = 5)
    fgmask = cv2.dilate(fgmask,(3,3),iterations = 5)
    cv2.imshow('frame',fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
