#!/usr/bin/python
import cv2
import rospy
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

class grayscaler:
    def __init__(self):
        self.bridge = CvBridge()
        self.color_sub_ = rospy.Subscriber('/camera/color/image_rect_color', Image, self.convert_callback)
        self.gray_pub_ = rospy.Publisher('/camera/gray/image_rect_gray', Image, queue_size=10)
    def convert_callback(self,msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        (rows,cols,channels) = cv_image.shape
        try:
            img_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            self.gray_pub_.publish(self.bridge.cv2_to_imgmsg(img_gray, "mono8"))
            # print "working"
        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    # Initialize Node
    rospy.init_node('greyscaler')

    # init path_manager_base object
    convert = grayscaler()

    while not rospy.is_shutdown():
        rospy.spin()
