import numpy as np
import cv2
import sys

if len(sys.argv) == 2:
    readfile = sys.argv[1]
else:
    print 'not enough input arguements so its gonna break'

cap = cv2.VideoCapture(readfile)

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',frame) #change second word to grey to playback in greyscale
    if cv2.waitKey(20) & 0xFF == ord('q'): #change waitkey to adjust the playback speed
        break

cap.release()
cv2.destroyAllWindows()
