import numpy as np
import cv2

cap = cv2.VideoCapture(1)

## this section was used for viewing image in greyscale
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    src = frame
    ksize = (15,15)
    sigmaX = 0
    dst = 'blurred'
    sigmaY = 1
    borderType = 'BORDER_TRANSPARENT'
    blur = cv2.GaussianBlur(src,ksize,sigmaX)#[,dst[,sigmaY[,borderType]]])
    # color = cv2.cvtColor(frame, cv2.COLOR)
    # Display the resulting frame
    cv2.imshow('GausBlur',frame)
    # cv2.imshow('frame')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
