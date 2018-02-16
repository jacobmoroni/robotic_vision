import cv2, numpy as np
from IPython.core.debugger import set_trace

meas=[]
pred=[]
measure=[]
frame = np.zeros((400,400,3), np.uint8) # drawing canvas
mp = np.array((2,1), np.float32) # measurement
move = np.array((2,1), np.float32) # measurement
tp = np.zeros((2,1), np.float32) # tracked / prediction
pose = np.array([[200],[200]],np.float32)

def walk():
    global move,measure,pose
    pose = pose + np.random.randn(2,1)*5
    move = np.array([[np.float32(np.rint(float(pose[0])))],[np.float32(np.rint(float(pose[1])))]])
    measure.append((pose[0],pose[1]))

def onmouse(k,x,y,s,p):
    global mp,meas
    mp = np.array([[np.float32(x)],[np.float32(y)]])
    meas.append((x,y))

# def paint():
#     global frame,meas,pred
#     for i in range(len(meas)-1): cv2.line(frame,meas[i],meas[i+1],(0,100,0))
#     for i in range(len(pred)-1): cv2.line(frame,pred[i],pred[i+1],(0,0,200))
def paint():
    global frame,measure,pred
    for i in range(len(measure)-1): cv2.line(frame,measure[i],measure[i+1],(0,100,0))
    for i in range(len(pred)-1): cv2.line(frame,pred[i],pred[i+1],(0,0,200))

def reset():
    global meas,pred,frame
    meas=[]
    pred=[]
    frame = np.zeros((400,400,3), np.uint8)

cv2.namedWindow("kalman")
cv2.setMouseCallback("kalman",onmouse);
kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03

#kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 0.00003
while True:
    walk()
    # kalman.correct(move)
    # kalman.correct(mp)
    tp = kalman.predict()
    kalman.correct(move)
    pred.append((int(tp[0]),int(tp[1])))
    paint()
    cv2.imshow("kalman",frame)
    k = cv2.waitKey(30) &0xFF
    if k == 27: break
    if k == 32: reset()
    # set_trace()
