#this removes python2.7 paths so it wont screw everything up with python3
import sys
# dir_remove = []
# for p in sys.path:
#     if p.find('python2') !=-1:
#         dir_remove.append(p)
# for p in dir_remove:
#     sys.path.remove(p)

import numpy as np
import cv2
# import video
from common import anorm2, draw_str
from time import clock
from kalman2d import Kalman2D
# from IPython.core.debugger import set_trace

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 30
        self.detect_interval = 1
        self.tracks = []
        # self.cam = video.create_capture(video_src)
        self.frame_idx = 0
        self.kf = Kalman2D()
        self.firsttime = True
        # self.cam = cv2.VideoCapture('mouse_tracking.mp4')
        self.cam = cv2.VideoCapture('videos/mv2_001.avi')
        # self.cam = cv2.VideoCapture(0)

    def selector(self):
        ret,frame = self.cam.read()
        self.r = cv2.selectROI(frame,False)
        self.prediction = np.reshape(np.array([[self.r[1]],[self.r[0]],[10],[10]],np.float32),4,1)
        # print (self.track_box)

    def run(self):
        self.selector()
        self.b_sub = []
        _ret, frame0 = self.cam.read()
        frame_gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        _ret, frame1 = self.cam.read()
        self.prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        img0,img1 = frame_gray0,self.prev_gray
        while True:
            _ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.b_sub = cv2.absdiff(img0,img1)
            ret1,self.b_sub = cv2.threshold(self.b_sub,50,255,cv2.THRESH_BINARY)
            self.b_sub = cv2.erode(self.b_sub,(3,3),iterations = 1)
            self.b_sub = cv2.dilate(self.b_sub,(3,3),iterations = 2)
            img0, img1 = self.prev_gray, frame_gray
            cv2.imshow('BackSub',self.b_sub)
            vis = frame.copy()
            if cv2.waitKey(1) & 0xFF == ord('p'):#Pause
                self.selector()
            if len(self.tracks) > 0:
                # img0, img1 = self.prev_gray, frame_gray
                # self.b_sub = cv2.absdiff(img0,img1)
                # ret1,self.b_sub = cv2.threshold(self.b_sub,50,255,cv2.THRESH_BINARY)
                # print self.b_sub
                # cv2.imshow('BackSub',self.b_sub)
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                diff = p1-p0
                moving1 = diff[:,0,0] > .25
                moving2 = diff[:,0,1] > .2
                moving = np.logical_or(moving1,moving2)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                good = np.logical_and(good, moving)
                new_tracks = []
                usum = 0
                vsum = 0
                umax = 0
                vmax = 0
                num_tracks = 0
                for tr, (x, y), (u,v), good_flag in zip(self.tracks, p1.reshape(-1, 2), diff.reshape(-1,2),good):
                    if not good_flag:
                        continue
                    num_tracks +=1
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    if u > umax:
                        umax = u
                    if v > vmax:
                        vmax = v
                    usum+=u
                    vsum+=v
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                if num_tracks > 0:
                    umean = usum/num_tracks
                    vmean = vsum/num_tracks
                else:
                    umean = 0
                    vmean = 0

                rtemp1 = self.r[1]+vmean
                rtemp3 = self.r[3]
                rtemp0 = self.r[0]+umean
                rtemp2 = self.r[2]
                xdot = vmean/30
                ydot = umean/30

                self.kf.measurement(rtemp1,rtemp0,xdot,ydot)
                self.kf.kalman.correct(self.kf.move)
                if self.firsttime == True:

                    while self.prediction[2] > .03 or abs(self.prediction[0]-self.r[1]) > 1:
                        self.kf.kalman.correct(self.kf.move)
                        self.prediction = self.kf.kalman.predict()
                        # print abs(self.prediction[0]-self.r[1])
                        # print self.prediction[2]
                    self.firsttime = False
                else:
                    self.prediction = self.kf.kalman.predict()
                    # self.prediciton = self.kf.kalman.correct(self.kf.move)

                self.kf.pred.append((int(self.prediction[0]),int(self.prediction[1])))
                # self.kf.paint(vis)
                rtemp1 = float(self.prediction[0])
                rtemp0 = float(self.prediction[1])
                # rtemp1 = self.r[1]+vmax
                # rtemp3 = self.r[3]
                # rtemp0 = self.r[0]+umax
                # rtemp2 = self.r[2]
                # print (self.r)
                self.r = (np.rint(rtemp0),np.rint(rtemp1),np.rint(rtemp2),np.rint(rtemp3))
                cv2.rectangle(vis,(int(self.r[0]),int(self.r[1])), (int(self.r[0])+int(self.r[2]),int(self.r[1])+int(self.r[3])), (0,0,255), 2)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                p_new = []
                if self.b_sub == []:
                    self.b_sub = np.zeros_like(frame_gray)
                    self.b_sub[:,:] = 255
                    # print self.b_sub
                mask = np.zeros_like(frame_gray)
                mask[int(self.r[1]):int(self.r[1]+self.r[3]), int(self.r[0]):int(self.r[0]+self.r[2])] = 255
                mask = np.logical_and(mask, self.b_sub).astype(np.uint8)*255


                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])


            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)

            ch = cv2.waitKey(1)
            if ch == 27:
                break

def main():
    # kf = Kalman2D()
    App(0).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
