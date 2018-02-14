#!/usr/bin/env python
'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
'''

# from __future__ import print_function
#this removes python2.7 paths so it wont screw everything up with python3
import sys
dir_remove = []
for p in sys.path:
    if p.find('python2') !=-1:
        dir_remove.append(p)
for p in dir_remove:
    sys.path.remove(p)

import numpy as np
import cv2
# import video
from common import anorm2, draw_str
from time import clock

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
        self.cam = cv2.VideoCapture('mv2_001.avi')
        ret,frame = self.cam.read()
        self.r = cv2.selectROI(frame,False)
        # print (self.track_box)

    def run(self):
        while True:
            _ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                diff = p1-p0
                moving1 = diff[:,0,0] > .1
                moving2 = diff[:,0,1] > .1
                moving = np.logical_or(moving1,moving2)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                good = np.logical_and(good, moving)
                new_tracks = []
                usum = 0
                vsum = 0
                num_tracks = 0
                for tr, (x, y), (u,v), good_flag in zip(self.tracks, p1.reshape(-1, 2), diff.reshape(-1,2),good):
                    if not good_flag:
                        continue
                    num_tracks +=1
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    usum+=u
                    vsum+=v
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                if num_tracks > 0:
                    umean = usum/num_tracks
                    vmean = vsum/num_tracks
                else:
                    umean = 0
                    vmean = 0
                # print (umean)

                rtemp1 = self.r[1]+vmean
                rtemp3 = self.r[3]
                rtemp0 = self.r[0]+umean
                rtemp2 = self.r[2]
                # print (self.r)
                self.r = (np.rint(rtemp0),np.rint(rtemp1),np.rint(rtemp2),np.rint(rtemp3))
                cv2.rectangle(vis,(int(self.r[0]),int(self.r[1])), (int(self.r[0])+int(self.r[2]),int(self.r[1])+int(self.r[3])), (0,0,255), 2)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                p_new = []
                mask = np.zeros_like(frame_gray)
                mask[int(self.r[1]):int(self.r[1]+self.r[3]), int(self.r[0]):int(self.r[0]+self.r[2])] = 255
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
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
