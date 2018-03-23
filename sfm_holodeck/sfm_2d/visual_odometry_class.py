import numpy as np
import cv2
from pdb import set_trace
import transforms3d

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_THIRD_FRAME = 2
STAGE_DEFAULT_FRAME = 3
kMinNumFeature = 1500

lk_params = dict(winSize = (21,21),
                    # maxLevel = 3
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

def featureTracking(image_ref,image_cur,px_ref):
    kp2,st,err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)

    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1,kp2

class PinholeCamera:
    def __init__(self,width,height,fx,fy,cx,cy,k1=0.0, k2 =0.0, p1=0.0,p2=0.0,k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1,k2,p1,p2,k3]


class VisualOdometry:
    def __init__(self,cam):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.intermediate_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx,cam.cy)
        self.trueX, self.trueY, self.trueZ = 0,0,0
        self.detector = cv2.FastFeatureDetector_create(threshold = 25, nonmaxSuppression = True)
        self.k_scale = .622 #this is the scale used for every other frame
        self.px_cur = np.array([0,0])
        # self.k_scale = .68
        # self.annotations = state_data

    def getAbsoluteScale(self, total_vel , positions):
        # ss = self.annotations[frame_id-1].strip().split()
        # print (positions)
        x_prev = float(positions[0][0])
        y_prev = float(positions[1][0])
        z_prev = float(positions[2][0])
        # ss = self.annotations[frame_id].strip().split()
        x = float(positions[0][2])
        y = float(positions[1][2])
        z = float(positions[2][2])
        self.trueX,self.trueY,self.trueZ = x,y,z
        scale = np.sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev))
        return self.k_scale*scale
        # return total_vel/30.0

    def processFirstFrame(self):
        ##This Section uses good features to track
        # mask = np.zeros_like(self.new_frame)
        # mask[:] = 255
        # self.px_ref = cv2.goodFeaturesToTrack(self.new_frame, mask = mask, **feature_params)
        # self.px_ref = np.squeeze(self.px_ref)

        ##This section uses FAST
        self.px_ref_old = self.detector.detect(self.new_frame)
        self.px_ref_old = np.array([x.pt for x in self.px_ref_old],dtype=np.float32)

        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        ##This Section uses good features to track
        # mask = np.zeros_like(self.new_frame)
        # mask[:] = 255
        # self.px_ref = cv2.goodFeaturesToTrack(self.new_frame, mask = mask, **feature_params)
        # self.px_ref = np.squeeze(self.px_ref)

        ##This section uses FAST
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref],dtype=np.float32)

        self.frame_stage = STAGE_THIRD_FRAME

    def processThirdFrame(self):
        self.px_ref_old,self.px_cur = featureTracking(self.last_frame,self.new_frame,self.px_ref_old)
        E,mask = cv2.findEssentialMat(self.px_cur,self.px_ref_old, focal = self.focal, pp=self.pp,method=cv2.RANSAC,prob=0.999,threshold=.3)
        _,self.cur_R,self.cur_t,mask = cv2.recoverPose(E,self.px_cur,self.px_ref_old,focal=self.focal,pp = self.pp)
        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref_old = self.px_ref
        self.px_ref = self.px_cur

    def processFrame(self, total_vel, positions):
        self.px_ref_old,self.px_cur = featureTracking(self.last_frame,self.new_frame,self.px_ref_old)
        E,mask = cv2.findEssentialMat(self.px_cur,self.px_ref_old, focal = self.focal, pp=self.pp,method=cv2.RANSAC,prob=0.999,threshold=.3)
        R1,R2,_ = cv2.decomposeEssentialMat(E)
        if np.trace(R1)>2.5:
            R_des = R1
        elif np.trace(R2)>2.5:
            R_des = R2
        else:
            print ("both are bad")
            R_des = np.array([[1,0,0],[0,1,0],[0,0,1]])
        R = R_des
        # print (R1)
        _,_,t,mask = cv2.recoverPose(E,self.px_cur,self.px_ref_old,focal=self.focal,pp = self.pp)
        # eulers = transforms3d.euler.mat2euler(R,'rxyz')
        # print (eulers[0])
        absolute_scale = self.getAbsoluteScale(total_vel,positions)
        if(absolute_scale > 0):
            self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)
        if(self.px_ref.shape[0] < kMinNumFeature):
            mask = np.zeros_like(self.new_frame)
            mask[:] = 255
            ##goodFeaturesToTrack
            # self.px_cur = cv2.goodFeaturesToTrack(self.new_frame, mask = mask, **feature_params)
            # self.px_cur = np.squeeze(self.px_cur)

            ##FAST
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
        self.px_ref_old = self.px_ref
        self.px_ref = self.px_cur

    def update(self,img,total_vel,positions):
        assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: Provided image isn't the same size as camera model or is not grayscale"
        self.new_frame = img
        if (self.frame_stage == STAGE_DEFAULT_FRAME):
            self.processFrame(total_vel,positions)
        elif (self.frame_stage == STAGE_THIRD_FRAME):
            self.processThirdFrame()
        elif (self.frame_stage == STAGE_SECOND_FRAME):
            self.processSecondFrame()
        elif (self.frame_stage == STAGE_FIRST_FRAME):
            self.processFirstFrame()
        self.last_frame = self.intermediate_frame
        self.intermediate_frame = self.new_frame
        # self.last_frame = self.new_frame
        P = np.hstack((self.cur_R,self.cur_t))
        print (self.px_cur)
        return self.px_cur,P
