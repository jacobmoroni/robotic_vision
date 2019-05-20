import numpy as np
import cv2

from visual_odometry_video import PinholeCamera, VisualOdometry

cam = PinholeCamera(1920.0,1080.0,718.8560,718.8560,1920.0/2,1080/2)
vo = VisualOdometry(cam)

traj = np.zeros((600,600,3),dtype=np.uint8)
cap = cv2.VideoCapture('videos/DJI_0065.MP4')
img_id = 1
while True:
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    vo.update(img, img_id)

    cur_t = vo.cur_t
    if (img_id > 2):
        x,y,z = cur_t[0],cur_t[1],cur_t[2]
    else:
        x,y,z = 0.,0.,0.
    draw_x,draw_y = int(x)+200,int(z)+90
    true_x,true_y = int(vo.trueX)+200,int(vo.trueZ)+90

    cv2.circle(traj, (draw_x,draw_y),1,(img_id*255/4540,255-img_id*255/4540,0),1)
    cv2.circle(traj, (true_x,true_y),1,(0,0,255),2)
    cv2.rectangle(traj,(10,20),(600,60),(0,0,0),-1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm" %(x,y,z)
    cv2.putText(traj,text,(20,40),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1,0)

    cv2.imshow("Road Facing Camera",img)
    cv2.imshow("Trajectory", traj)
    img_id += 1
    cv2.waitKey(1)

cv2.imwrite('map.png',traj)
