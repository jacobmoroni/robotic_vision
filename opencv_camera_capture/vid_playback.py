import numpy as np
import cv2

cap = cv2.VideoCapture('output.avi')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',frame) #change second word to grey to playback in greyscale
    if cv2.waitKey(20) & 0xFF == ord('q'): #change waitkey to adjust the playback speed
        break

cap.release()
cv2.destroyAllWindows()
