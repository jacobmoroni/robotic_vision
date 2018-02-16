import cv2, numpy as np

class Kalman2D(object):
    """docstring forKalman2D."""
    def __init__(self):
        self.meas=[]
        self.pred=[]
        self.measure=[]
        # self.frame = np.zeros((400,400,3), np.uint8) # drawing canvas
        self.move = np.array((2,1), np.float32) # measurement
        self.tp = np.zeros((2,1), np.float32) # tracked / prediction
        Ts = np.float32(1/30)
        # self.kalman = cv2.KalmanFilter(4,4) #for measuring x, y, only
        self.kalman = cv2.KalmanFilter(4,4) #for measuring x, y, xdot,ydot

        # self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32) #for measuring x and y only
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) #for measuring x, y, xdot,ydot

        # self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,Ts,0],[0,1,0,Ts],[0,0,1,0],[0,0,0,1]],np.float32)

        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * .01
        # self.kalman.processNoiseCov = np.array([[1/3*Ts**3,0,1/2*Ts**2,0],
        #                                         [0,1/3*Ts**3,0,1/2*Ts**2],
        #                                         [1/2*Ts**2,0,Ts,0],
        #                                         [0,1/2*Ts**2,0,Ts]],np.float32) * 1e2

        # self.kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 1e-5 #for measuring x and y only
        self.kalman.measurementNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 1e-5  #for measuring x, y, xdot,ydot

    # def measurement(self,x,y): #for measuring x and y only
    #     self.move = np.reshape(np.array([[x],[y]],np.float32),2,1)
    #     self.measure.append((x,y))

    def measurement(self,x,y,xdot,ydot):
        self.move = np.reshape(np.array([[x],[y],[xdot],[ydot]],np.float32),4,1)
        self.measure.append((x,y))

    def paint(self,frame):
        for i in range(len(self.measure)-1): cv2.line(frame,self.measure[i],self.measure[i+1],(0,100,0))
        for i in range(len(self.pred)-1): cv2.line(frame,self.pred[i],self.pred[i+1],(0,0,200))
