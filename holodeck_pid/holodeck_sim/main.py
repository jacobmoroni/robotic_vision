# This is for problems with clashing opencv versions from ROS installations
import sys
rm_python2 = []
for p in sys.path:
    if p.find('python2') != -1:
        rm_python2.append(p)
for p in rm_python2:
    sys.path.remove(p)

import cv2
import numpy as np
import math
import time

from uav_sim import UAVSim
from multi_image import MultiImage

urban_world = 'UrbanCity'
forest_world = 'EuropeanForest'
redwood_world = 'RedwoodForest'

edge_min = 150
edge_max = 200

def holodeck_sim():
    uav_sim = UAVSim(urban_world)
    uav_sim.init_teleop()
    uav_sim.init_plots(plotting_freq=5) # Commenting this line would disable plotting
    uav_sim.command_velocity = True # This tells the teleop to command velocities rather than angles

    multi_img = MultiImage(2,2)

    while True:
        # This is the main loop where the simulation is updated
        uav_sim.step_sim()
        cam = uav_sim.get_camera()

        # I run my opencv stuff here
        gray = cv2.cvtColor(cam, cv2.COLOR_RGBA2GRAY)
        edge = cv2.Canny(cam, edge_min, edge_max)
        bgr = cv2.cvtColor(cam, cv2.COLOR_RGBA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # You can also give an external command and switch between automatic
        # commands given here and manual commands from the keyboard using the
        # key mapped to MANUAL_TOGGLE in uav_sim.py

        # In automatic mode, fly forward at 3m altitude at current heading
        yaw_c = uav_sim.yaw_c
        uav_sim.command_velocity(vx=2.0, vy=0.0, yaw=yaw_c, 3.0)

        # This is just a useful class for viewing multiple filters in one image
        multi_img.add_image(cam, 0,0)
        multi_img.add_image(gray, 0,1)
        multi_img.add_image(edge, 1,0)
        multi_img.add_image(hsv, 1,1)
        display = multi_img.get_display()
        cv2.imshow('Holodeck', display)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    holodeck_sim()
    print("Finished")
