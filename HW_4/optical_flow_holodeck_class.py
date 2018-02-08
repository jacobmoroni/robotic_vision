#this removes python2.7 paths so it wont screw everything up
import sys
dir_remove = []
for p in sys.path:
    if p.find('python2') !=-1:
        dir_remove.append(p)
for p in dir_remove:
    sys.path.remove(p)

#other import commands
import numpy as np
from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
import pygame
from pygame.locals import *
import cv2
import time
import scipy.io as sio
import transforms3d
from common import anorm2, draw_str
from time import clock
# import holodeck_functions
cv2.namedWindow('lk_track')
# determine whether the ouput should be saved
if len(sys.argv) ==2:
    outfile = sys.argv[1]
    savefile = True
else:
    savefile = False

display_width = 400
display_height = 100
black = (0,0,0)
white = (255,255,255)
red = (255,0,0)

#initialize pygame window
pygame.display.init()
default_font = pygame.font.get_default_font()
gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Holodeck")
clock = pygame.time.Clock()
pygame.font.init()

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class holodeck_fly:
    """docstring for holodeck_fly."""
    def __init__(self):
        self.track_len = 5
        self.detect_interval = 1
        self.tracks = []
        self.frame_idx = 0
        # self.cam = cv2.VideoCapture(0)

        self.running = True

        #start Holodeck
        self.env = Holodeck.make("UrbanCity")

        self.state_mat = {'x_c':[],'y_c':[],'u_c':[],'v_c':[],'yaw_c':[],
                     'roll_c':[],'pitch_c':[],'yaw_rate_c':[],'alt_c':[],
                     'roll':[],'pitch':[],'yaw':[],
                     'x':[],'y':[],'z':[],
                     'u':[],'v':[],'w':[],
                     'p':[],'q':[],'r':[],
                     'ax':[],'ay':[],'az':[]}

        self.u_c = 0
        self.v_c = 0
        self.yaw_c = 0
        self.altitude_c = 0
        self.optical = 1
        self.position_command = 1

        self.ALTITUDE_UP = K_w
        self.ALTITUDE_DOWN = K_s
        self.YAW_CCW = K_a
        self.YAW_CW = K_d
        self.FWD_VEL = K_UP
        self.BACK_VEL = K_DOWN
        self.RIGHT_VEL = K_RIGHT
        self.LEFT_VEL = K_LEFT
        self.OPTIC = K_o
        self.POSITION_COMMAND = K_1
        self.QUIT = K_ESCAPE

        self.firstframe= True
        self.firsttime = True
        self.location_c = [-6.818,-30,1.6,5.0] #x,y -11.455,psi 1.6,h        # super(holodeck_fly, self).__init__()
        # self.canyon_area_l = 100:410,0:200
        p_grid = np.array([[0,0]])
        for i in range(0,11):
            for j in range(0,11):
                p_new = np.array([[30+45*i,30+45*j]])
                # print (p_new)
                p_grid = np.concatenate((p_grid,p_new))
        self.p_grid = p_grid[1:p_grid.shape[0]+1]

        # self.prev_gray_4 = []
        self.prev_gray_3 = []
        self.prev_gray_2 = []
        self.prev_gray = []

    def unwrap(self, angle):
        while angle >= np.pi:
            angle = angle-2*np.pi
        while angle < -np.pi:
            angle = angle+2*np.pi
        return angle

    def saturate(self,sat_input,sat):
        if sat_input > sat:
            sat_output = sat
        elif sat_input < -sat:
            sat_output = -sat
        else:
            sat_output = sat_input
        return sat_output

    def text_objects(self, text, font):
        textSurface = font.render(text, True,white)
        return textSurface, textSurface.get_rect()

    def message_display(self, command):
        alt_str = str(round(command[3],2))
        yaw_str = str(round(command[2],2))
        pitch_str = str(round(command[1],2))
        roll_str = str(round(command[0],2))
        comma = str(' , ')
        command_text = str(roll_str+comma+pitch_str+comma+yaw_str+comma+alt_str)
        text = str(command_text)

        gameDisplay.fill(black)
        largeText = pygame.font.Font(default_font,30)
        TextSurf, TextRect = self.text_objects(text, largeText)
        TextRect.center = ((display_width/2),(display_height/2))
        gameDisplay.blit(TextSurf, TextRect)

        pygame.display.update()

    def controller(self,command_input,body_velocity,eulers,imu):
        q = imu[4]
        p = imu[3]
        r = imu[5]
        # k_p_phi = .2
        # k_p_theta = .2
        k_p_phi = .2
        k_p_theta = .2
        k_p_yaw = 3
        k_d_phi = .1
        k_d_theta = .1
        k_d_yaw = .5
        u = body_velocity[0]
        v = body_velocity[1]
        yaw = eulers[2]
        self.u_c = command_input[0]
        self.v_c = command_input[1]
        self.yaw_c = command_input[2]
        h_c = command_input[3]
        phi_c = -k_p_phi*(self.v_c-v) + k_d_phi*(q)
        yaw_diff = self.yaw_c-yaw
        self.yaw_c = self.unwrap(self.yaw_c)
        yaw_diff = self.unwrap(yaw_diff)
        yaw_rate_c = k_p_yaw*(yaw_diff) + k_d_yaw*(r)
        theta_c = -k_p_theta*(self.u_c-u) + k_d_theta*(r)
        phi_c = self.saturate(phi_c,.6)
        theta_c = self.saturate(theta_c,.6)
        vel_command = np.array([phi_c,theta_c,yaw_rate_c,h_c])
        return vel_command

    def position_controller(self,location,yaw):
        k_u = 1;
        k_v = 1;

        x_c = self.location_c[0]
        y_c = self.location_c[1]

        yaw_c = self.location_c[2]
        h_c = self.location_c[3]
        yaw_diff = yaw_c-yaw
        x = location[0]
        y = location[1]
        x_diff = x_c-x
        y_diff = y_c-y

        pn_diff = self.saturate(x_diff,2)
        pe_diff = self.saturate(y_diff,2)
        p_diff = np.array([[float(pn_diff)],[float(pe_diff)]])

        rot_2d = np.array([[np.cos(yaw), -np.sin(yaw)],
                           [np.sin(yaw), np.cos(yaw)]])

        x_diff,y_diff = np.matmul(rot_2d,p_diff)

        u_c = k_u*(x_diff)
        v_c = k_v*(y_diff)

        return u_c,v_c,yaw_c,h_c

    def uav_teleop(self,command):
        state, reward, terminal, _ = self.env.step(command)
        alt_str = str(command[3])
        yaw_str = str(command[2])
        pitch_str = str(command[1])
        roll_str = str(command[0])

        command_str = str(command)
        self.message_display(command)
        return state

    def update_state_mat(self, command_output,command_input,eulers,location,body_vel,velocity,imu):
        self.state_mat['x_c'].append(self.location_c[0])
        self.state_mat['y_c'].append(self.location_c[1])

        self.state_mat['u_c'].append(command_input[0])
        self.state_mat['v_c'].append(command_input[1])
        self.state_mat['yaw_c'].append(command_input[2])
        self.state_mat['roll_c'].append(command_output[0])
        self.state_mat['pitch_c'].append(command_output[1])
        self.state_mat['yaw_rate_c'].append(command_output[2])
        self.state_mat['alt_c'].append(command_output[3])

        self.state_mat['roll'].append(eulers[0])
        self.state_mat['pitch'].append(eulers[1])
        self.state_mat['yaw'].append(eulers[2])

        location = location
        self.state_mat['x'].append(location[0])
        self.state_mat['y'].append(location[1])
        self.state_mat['z'].append(location[2])

        imu = imu
        self.state_mat['p'].append(imu[3].copy())
        self.state_mat['q'].append(imu[4].copy())
        self.state_mat['r'].append(imu[5].copy())
        self.state_mat['ax'].append(imu[0].copy())
        self.state_mat['ay'].append(imu[1].copy())
        self.state_mat['az'].append(imu[2].copy())

        self.state_mat['u'].append(body_vel[0])
        self.state_mat['v'].append(body_vel[1])
        self.state_mat['w'].append(velocity[2])

    def fetch_keys(self):
        keys = pygame.key.get_pressed()
        #Mixer that sends keys pressed to commands
        if keys[self.ALTITUDE_UP]:
            self.altitude_c += .5
        if keys[self.ALTITUDE_DOWN]:
            if self.altitude_c >0:
                self.altitude_c -= .5
            else:
                self.altitude_c =0
        else:
            self.altitude_c =  self.altitude_c
        if keys[self.YAW_CW]:
            self.yaw_c -= .05
        elif keys[self.YAW_CCW]:
            self.yaw_c += .05
        else:
            self.yaw_c = self.yaw_c
        if keys[self.FWD_VEL]:
            self.u_c = 5
        elif keys[self.BACK_VEL]:
            self.u_c = -5
        else:
            self.u_c = 0
        if keys[self.LEFT_VEL]:
            self.v_c = -5
        elif keys[self.RIGHT_VEL]:
            self.v_c = 5
        else:
            self.v_c = 0
        if keys[self.QUIT]:
            self.running = False
        if keys[self.OPTIC]:
            self.optical +=1
        if keys[self.POSITION_COMMAND]:
            self.position_command +=1
        # return self.u_c,self.v_c,self.yaw_c,self.altitude_c,self.running

    def run(self):
        while self.running == True :
            pygame.event.pump() #this clears events to make sure fresh ones come in
            self.fetch_keys()

            if self.position_command %2 ==0:
                u_c,v_c,yaw_c,h_c = self.position_controller(location,eulers[2])
                command_input = np.array([u_c,v_c,yaw_c,h_c])
                self.altitude_c = h_c
                # print (command_input)
            else:
                command_input = np.array([self.u_c,self.v_c,self.yaw_c,self.altitude_c])

            if self.firsttime == True:
                state = self.uav_teleop([0,0,0,0])
                self.firsttime = False
            else:
                state = self.uav_teleop(command_output)

            #state extracts the sensor information
            pixels = state[Sensors.PRIMARY_PLAYER_CAMERA]
            orientation = state[Sensors.ORIENTATION_SENSOR]
            location = state[Sensors.LOCATION_SENSOR]
            location = location/100
            velocity = state[Sensors.VELOCITY_SENSOR]
            velocity = velocity/100
            imu = state[Sensors.IMU_SENSOR]

            eulers = transforms3d.euler.mat2euler(orientation,'rxyz')
            rot_2d = np.array([[np.cos(eulers[2]), -np.sin(eulers[2])],
                               [np.sin(eulers[2]), np.cos(eulers[2])]])
            planar_vel = np.array([velocity[0],velocity[1]])
            body_vel= np.matmul(rot_2d,planar_vel)
            # body_vel= body_vel/100
            angle_command = self.controller(command_input,body_vel,eulers,imu)
            roll_c = float(angle_command[0])
            pitch_c = float(angle_command[1])
            yaw_rate_c = float(angle_command[2])
            h_c = float(angle_command[3])
            command_output = np.array([roll_c,pitch_c, yaw_rate_c, h_c])
            # print (location,+eulers[2])

            self.update_state_mat(command_output,command_input,eulers,location,body_vel,velocity,imu)

            if self.optical%2 == 0:
                frame = pixels

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis = frame.copy()

                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray_2, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    d = abs(p0-p0r).reshape(-1, 2).max(-1)

                    #find pixel velocities
                    diff = p1-p0
                    canyon_left = []
                    canyon_right = []
                    obstacle = []
                    alt_vel = []
                    for i, x in enumerate(self.p_grid):
                        if 100 < x[0] < 410 and x[1] < 185:
                            canyon_left.append(i)
                        if 100 < x[0] < 410 and 325 < x[1] < 510:
                            canyon_right.append(i)
                        if 140 < x[0] < 370 and 140 < x[1] < 370:
                            obstacle.append(i)
                        if 325 < x[0] < 510 and 140 < x[1] < 370:
                            alt_vel.append(i)
                    canyon_left = np.array([canyon_left])
                    canyon_right = np.array([canyon_right])
                    obstacle = np.array([obstacle])
                    alt_vel = np.array([alt_vel])
                    canyon_left_vel = []
                    for num in canyon_left:
                        canyon_left_vel.append(diff[num][0])
                    canyon_right_vel = []
                    for num in canyon_right:
                        canyon_right_vel.append(diff[num][0])
                    alt_vel_vel = []
                    for num in alt_vel:
                        alt_vel_vel.append(diff[num][0])
                    canyon_left_vel_sum = sum(canyon_left_vel)
                    canyon_right_vel_sum = sum(canyon_right_vel)
                    alt_vel_vel_sum = sum(alt_vel_vel)
                    # print (canyon_left_vel_sum)
                    # print (np.shape(canyon_left_vel_sum))
                    cv2.arrowedLine(vis, (92,255), (92+int(20*canyon_left_vel_sum[0][0]),255), (0,255,0), 2) #int(20*canyon_left_vel_sum[0][0]))
                    cv2.arrowedLine(vis, (417,255), (417+int(20*canyon_right_vel_sum[0][0]),255), (0,255,0), 2) #+int(20*canyon_right_vel_sum[0][0])

                    # good = d < 1
                    good = d
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

                    #plot foe and search areas
                    cv2.circle(vis,(255,255),5,(0,0,255),1)
                    cv2.rectangle(vis,(0,100),(185,410), (0,255,0), 1)
                    cv2.rectangle(vis,(325,100),(510,410), (0,255,0), 1)
                    cv2.rectangle(vis,(140,140),(370,370), (255,0,0), 1)
                    cv2.rectangle(vis,(140,325),(370,510), (0,0,255), 1)
                    self.tracks = new_tracks
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                    draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
                if self.frame_idx % self.detect_interval == 0 and self.frame_idx>5:
                    self.tracks = []
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv2.circle(mask, (x, y), 5, 0, -1)
                    # p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                    p = self.p_grid
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])


                self.frame_idx += 1
                self.prev_gray_4 = self.prev_gray_3
                self.prev_gray_3 = self.prev_gray_2
                self.prev_gray_2 = self.prev_gray
                self.prev_gray = frame_gray

                cv2.imshow('lk_track', vis)

            if savefile == True:
                sio.savemat(outfile,self.state_mat)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def main():
    fly = holodeck_fly
    cv2.destroyAllWindows()

if __name__ == '__main__':
    fly = holodeck_fly()
    fly.run()
    cv2.destroyAllWindows
