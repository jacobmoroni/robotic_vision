import numpy as np
import cv2

# cap = cv2.VideoCapture("rtsp://184.72.239.149/vod/mp4:BigBuckBunny_115k.mov")
# cap = cv2.VideoCapture("rtsp://192.168.1.42:5000/out.h264")
# cap = cv2.VideoCapture('rtsp://192.168.1.2:8080/out.h264')
cap = cv2.VideoCapture("rtsp://192.168.1.42/onvif1")
# cap.open("http://192.168.226.101:8080/video?x.mjpeg")
## this section was used for viewing image in greyscale
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print ret
    # print (frame)
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # color = cv2.cvtColor(frame, cv2.COLOR)
    # Display the resulting frame
    if ret == True:
        cv2.imshow('frame',frame)
    # cv2.imshow('frame')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
