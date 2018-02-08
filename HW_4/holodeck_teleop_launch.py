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

# determine whether the ouput should be saved
if len(sys.argv) ==2:
    outfile = sys.argv[1]
    savefile = True
else:
    savefile = False

#initialize sliders for filters
def nothing(x):
    pass
cv2.namedWindow('image')

#initialize pygame window
display_width = 400
display_height = 100
black = (0,0,0)
white = (255,255,255)
red = (255,0,0)

pygame.display.init()
default_font = pygame.font.get_default_font()
gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Holodeck")
clock = pygame.time.Clock()
pygame.font.init()

running = True

#start Holodeck
env = Holodeck.make("UrbanCity")

state_mat = {'u_c':[],'v_c':[],'yaw_c':[],'roll_c':[],'roll':[],'pitch_c':[],'pitch':[],'yaw_rate_c':[],'yaw':[],'alt_c':[],'alt':[],'x':[],'y':[],'z':[],'u':[],'v':[],'w':[],
             'p':[],'q':[],'r':[],'ax':[],'ay':[],'az':[]}

u_c = 0
v_c = 0
yaw_c = 0
altitude_c = 0
ALTITUDE_UP = K_w
ALTITUDE_DOWN = K_s
YAW_CCW = K_a
YAW_CW = K_d

FWD_VEL = K_UP
BACK_VEL = K_DOWN
RIGHT_VEL = K_RIGHT
LEFT_VEL = K_LEFT

QUIT = K_ESCAPE
firstframe=True
firsttime = True

def unwrap(angle):
    while angle >= 2*np.pi:
        angle = angle-2*np.pi
    while angle < -2*np.pi:
        angle = angle+2*np.pi
    return angle

def text_objects(text, font):
    textSurface = font.render(text, True,white)
    return textSurface, textSurface.get_rect()

def message_display(command):
    alt_str = str(round(command[3],2))
    yaw_str = str(round(command[2],2))
    pitch_str = str(round(command[1],2))
    roll_str = str(round(command[0],2))
    comma = str(' , ')
    command_text = str(roll_str+comma+pitch_str+comma+yaw_str+comma+alt_str)
    text = str(command_text)

    gameDisplay.fill(black)
    largeText = pygame.font.Font(default_font,30)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((display_width/2),(display_height/2))
    gameDisplay.blit(TextSurf, TextRect)

    pygame.display.update()

def controller(command_input,body_velocity,eulers):
    k1 = .2
    k2 = .2
    k3 = 1
    u = body_velocity[0]
    v = body_velocity[1]
    yaw = eulers[2]
    u_c = -command_input[0]
    v_c = -command_input[1]
    yaw_c = command_input[2]
    phi_c = -k1*(v_c-v)
    yaw_diff = yaw_c-yaw
    print (yaw_c)
    print (yaw_diff)
    yaw_c = unwrap(yaw_c)
    yaw_diff = unwrap(yaw_diff)
    print (yaw_c)
    print (yaw_diff)
    yaw_rate_c = k3*(yaw_diff)
    print (u)
    theta_c = -k2*(u_c-u)
    if phi_c > 1:
        phi_c = 1
    elif phi_c < -1:
        phi_c = -1
    if theta_c >1:
        theta_c = 1
    elif theta_c < -1:
        theta_c = -1
    vel_command = np.array([phi_c,theta_c,yaw_rate_c])
    return vel_command

def uav_teleop(command):
    state, reward, terminal, _ = env.step(command)
    alt_str = str(command[3])
    yaw_str = str(command[2])
    pitch_str = str(command[1])
    roll_str = str(command[0])

    command_str = str(command)
    message_display(command)
    return state

def update_state_mat(command_output,command_input,eulers,location,body_vel,velocity,imu):
    state_mat['u_c'].append(-command_input[0])
    state_mat['v_c'].append(-command_input[1])
    state_mat['yaw_c'].append(command_input[2])
    state_mat['roll_c'].append(command_output[0])
    state_mat['pitch_c'].append(command_output[1])
    state_mat['yaw_rate_c'].append(command_output[2])
    state_mat['alt_c'].append(command_output[3])

    state_mat['roll'].append(eulers[0])
    state_mat['pitch'].append(eulers[1])
    state_mat['yaw'].append(eulers[2])

    location = location/100
    state_mat['x'].append(location[0])
    state_mat['y'].append(location[1])
    state_mat['z'].append(location[2])

    imu = imu
    state_mat['p'].append(imu[3].copy())
    state_mat['q'].append(imu[4].copy())
    state_mat['r'].append(imu[5].copy())
    state_mat['ax'].append(imu[0].copy())
    state_mat['ay'].append(imu[1].copy())
    state_mat['az'].append(imu[2].copy())

    state_mat['u'].append(body_vel[0])
    state_mat['v'].append(body_vel[1])
    state_mat['w'].append(velocity[2])

while running:

    pygame.event.pump() #this clears events to make sure fresh ones come in
    keys = pygame.key.get_pressed() #this looks for keys that are pressed

    #Mixer that sends keys pressed to commands
    if keys[ALTITUDE_UP]:
        altitude_c += .5
    if keys[ALTITUDE_DOWN]:
        if altitude_c >0:
            altitude_c -= .5
        else:
            altitude_c =0
    if keys[YAW_CW]:
        yaw_c -= .05
    elif keys[YAW_CCW]:
        yaw_c += .05
    else:
        yaw_rate_c = 0
    if keys[FWD_VEL]:
        u_c = -5
    elif keys[BACK_VEL]:
        u_c = 5
    else:
        u_c = 0
    if keys[LEFT_VEL]:
        v_c = 5
    elif keys[RIGHT_VEL]:
        v_c = -5
    else:
        v_c = 0

    command_input = np.array([u_c,v_c,yaw_c,altitude_c])

    if firsttime == True:
        state = uav_teleop([0,0,0,0])
        firsttime = False
    else:
        state = uav_teleop(command_output)

    #state extracts the sensor information
    pixels = state[Sensors.PRIMARY_PLAYER_CAMERA]
    orientation = state[Sensors.ORIENTATION_SENSOR]
    location = state[Sensors.LOCATION_SENSOR]
    velocity = state[Sensors.VELOCITY_SENSOR]
    imu = state[Sensors.IMU_SENSOR]

    eulers = transforms3d.euler.mat2euler(orientation,'rxyz')
    rot_2d = np.array([[np.cos(eulers[2]), -np.sin(eulers[2])],
                       [np.sin(eulers[2]), np.cos(eulers[2])]])
    planar_vel = np.array([velocity[0],velocity[1]])
    body_vel= np.matmul(rot_2d,planar_vel)
    body_vel= body_vel/100
    angle_command = controller(command_input,body_vel,eulers)
    roll_c = float(angle_command[0])
    pitch_c = float(angle_command[1])
    yaw_rate_c = float(angle_command[2])
    command_output = np.array([roll_c,pitch_c, yaw_rate_c, altitude_c])

    update_state_mat(command_output,command_input,eulers,location,body_vel,velocity,imu)

    # #the next block is looks for slider positions to determine filters
    frame = pixels
    if firstframe == True:
        src = frame
        firstframe=False

    cv2.imshow('image',src)


    if savefile == True:
        sio.savemat(outfile,state_mat)


    if keys[QUIT]:
        running = False
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows
