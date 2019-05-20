import numpy as np
import cv2

cap = cv2.VideoCapture(0)
# cap2 = cv2.VideoCapture(1)
cap3 = cv2.VideoCapture(2)
cap4 = cv2.VideoCapture(3)
# cap5 = cv2.VideoCapture(1)


## this section was used for viewing image in greyscale
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    # ret5, frame5 = cap5.read()
    # print frame5
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # color = cv2.cvtColor(frame, cv2.COLOR)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('frame3',frame3)
    cv2.imshow('frame4',frame4)
    # cv2.imshow('frame5',frame5)
    # cv2.imshow('frame')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
