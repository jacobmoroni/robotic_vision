import numpy as np
import cv2

cap = cv2.VideoCapture(0)

## this section was used for viewing image in greyscale
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # color = cv2.cvtColor(frame, cv2.COLOR)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    # cv2.imshow('frame')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
