import numpy as np
import cv2
from pdb import set_trace

cap = cv2.VideoCapture(0)
# take first frame of the video
ret,frame = cap.read()
frame = cv2.flip(frame,1)
#initialize mog2
fgbg = cv2.createBackgroundSubtractorMOG2(history = 2,detectShadows = False)

#initialize kalman filter
move = np.array((2,1), np.float32) # measurement
tp = np.zeros((2,1), np.float32) # tracked / prediction
Ts = np.float32(1/30)
kalman = cv2.KalmanFilter(4,2) #for measuring x, y, xdot,ydot
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32) #for measuring x, y, xdot,ydot
kalman.transitionMatrix = np.array([[1,0,Ts,0],[0,1,0,Ts],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * .01
kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 1e-5
# setup initial location of window
def selector():
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    rval = cv2.selectROI(frame,False)
    return rval

rval = selector()
r,h,c,w = rval[1],rval[3],rval[0],rval[2]
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
# bsub_roi = fgmask[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#this mask is for skin color
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

# set_trace()
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
    ret ,frame = cap.read()
    frame = cv2.flip(frame,1)
    if ret == True:
        fgmask = fgbg.apply(frame)
        fgmask = cv2.erode(fgmask,(3,3),iterations = 5)
        fgmask = cv2.dilate(fgmask,(3,3),iterations = 5)
        bsub_roi = fgmask[r:r+h, c:c+w]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        x_meas = track_window[0]+track_window[2]/2
        y_meas = track_window[1]+track_window[3]/2
        measurement = np.reshape(np.array([[x_meas],[y_meas]],np.float32),2,1)
        kalman.predict()
        prediction = kalman.correct(measurement)
        # set_trace()
        x_window = int(round(prediction[0]-track_window[2]/2.0))
        y_window = int(round(prediction[1]-track_window[3]/2.0))
        w_window = int(round(track_window[2]))
        h_window = int(round(track_window[3]))
        track_window = (x_window,y_window,w_window,h_window)
        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        cv2.imshow('img2',img2)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)
    else:
        break
cv2.destroyAllWindows()
cap.release()
