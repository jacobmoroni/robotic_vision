#sfm class
import numpy as np
import cv2
from pdb import set_trace
import transforms3d

class SfM_2d:
    def __init__(self):
        self.var = 1

    def triangulate(P1,P2,points1,points2):
        points_3d  = cv2.triangulatePoints(P1, P2, points1, points2)
        points_normed = points_3d/points_3d[3,:]
        return points_normed
