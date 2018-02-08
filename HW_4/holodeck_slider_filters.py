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
# colorvid = False
# if len(sys.argv) ==2:
#     outfile = sys.argv[1]
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(outfile,fourcc, 20.0, (640,480),colorvid)
#     savefile = True
if len(sys.argv) ==2:
    outfile = sys.argv[1]
    savefile = True
else:
    savefile = False

#initialize sliders for filters
def nothing(x):
    pass
cv2.namedWindow('image')
# create trackbars for color change
switch0 = 'greyscale (on/off)'
cv2.createTrackbar(switch0,'image',0,1,nothing)

switch = 'GausBlur (on/off)'
cv2.createTrackbar(switch, 'image',0,1,nothing)
cv2.createTrackbar('ksize H','image',1,300,nothing)
cv2.createTrackbar('ksize V','image',1,300,nothing)

switch2 = 'EdgeDetect (on/off)'
cv2.createTrackbar(switch2, 'image',0,1,nothing)
cv2.createTrackbar('minval','image',0,300,nothing)
cv2.createTrackbar('maxval','image',0,500,nothing)

switch3 = 'BilateralBlur (on/off)'
cv2.createTrackbar(switch3, 'image',0,1,nothing)
cv2.createTrackbar('diameter','image',0,100,nothing)
cv2.createTrackbar('sigmaColor','image',0,500,nothing)
cv2.createTrackbar('sigmaSpace','image',0,500,nothing)


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
def text_objects(text, font):
    textSurface = font.render(text, True,white)
    return textSurface, textSurface.get_rect()
def message_display(command):
    alt_str = str(command[3])
    yaw_str = str(command[2])
    pitch_str = str(command[1])
    roll_str = str(command[0])
    text = str(command)
    # text[0] = 'Roll: ' + roll_str
    # text[1] = 'Pitch: '+ pitch_str
    # text[2] = 'Yaw: '+ yaw_str
    # text[3] = 'Altitude: '+alt_str

    gameDisplay.fill(black)
    largeText = pygame.font.Font(default_font,30)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((display_width/2),(display_height/2))
    gameDisplay.blit(TextSurf, TextRect)

    pygame.display.update()
running = True
#start Holodeck
env = Holodeck.make("UrbanCity")
def uav_teleop(command):
    # env = Holodeck.make("UrbanCity")
    state, reward, terminal, _ = env.step(command)
    alt_str = str(command[3])
    yaw_str = str(command[2])
    pitch_str = str(command[1])
    roll_str = str(command[0])

    command_str = str(command)
    message_display(command)
    return state
def filters(firstframe,frame):
    #the next block is looks for slider positions to determine filters
    # frame = pixels
    if firstframe == True:
        src = frame
        firstframe=False

    cv2.imshow('image',src)
    # get current positions of four trackbars
    if cv2.getTrackbarPos('ksize H','image') %2 == 1:
        ksizeH = cv2.getTrackbarPos('ksize H','image')
        print (ksizeH)
    else:
        ksizeH = cv2.getTrackbarPos('ksize H','image')+1
    if cv2.getTrackbarPos('ksize H','image') <=0:
        ksizeH = 1;
    if cv2.getTrackbarPos('ksize V','image') %2 == 1:
        ksizeV = cv2.getTrackbarPos('ksize V','image')
    else:
        ksizeV = cv2.getTrackbarPos('ksize V','image')+1
    if cv2.getTrackbarPos('ksize V','image') <=0:
        ksizeV = 1;
    minval = cv2.getTrackbarPos('minval','image')
    maxval = cv2.getTrackbarPos('maxval','image')
    s0 = cv2.getTrackbarPos(switch0,'image')
    s = cv2.getTrackbarPos(switch,'image')
    s2 = cv2.getTrackbarPos(switch2,'image')
    s3 = cv2.getTrackbarPos(switch3,'image')
    ksize = (ksizeH,ksizeV)

    dia = cv2.getTrackbarPos('diameter','image')
    sigCol = cv2.getTrackbarPos('sigmaColor','image')
    sigSpace = cv2.getTrackbarPos('sigmaSpace','image')

    if s0 ==1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bilsrc = frame
    if s == 0 and s2 ==0 and s3 == 0:
        src = frame
        edgesrc = frame
        bilsrc = frame
    elif s == 0 and s2 == 1 and s3 == 0:
        src = edges
        edgesrc = frame
        bilsrc = frame
    elif s == 1 and s2 == 0 and s3 == 0:
        src = blur
        edgesrc = frame
        bilsrc = frame
    elif s == 1 and s2 == 1 and s3 == 0:
        src = edges
        edgesrc = blur
        bilsrc = frame
    elif s == 0 and s2 == 0 and s3 == 1:
        src = bilblur
        edgesrc = frame
        bilsrc = frame

    elif s == 0 and s2 == 1 and s3 == 1:
        src = edges
        edgesrc = bilblur
        bilsrc = frame
    else:
        src = edges
        edgesrc = blur
        bilsrc = frame

    blursrc = frame
    sigmaX = 0

    blur = cv2.GaussianBlur(blursrc,ksize,sigmaX)#[,dst[,sigmaY[,borderType]]])
    # bilblur = cv2.bilateralFilter(bilsrc,dia,sigCol,sigSpace)
    edges = cv2.Canny(edgesrc, minval, maxval) #doesnt work
def update_state_mat(command,eulers,location,velocity,imu):
    state_mat['roll_c'].append(command[0])
    state_mat['pitch_c'].append(command[1])
    state_mat['yaw_c'].append(command[2])
    state_mat['alt_c'].append(command[3])

    state_mat['roll'].append(eulers[0])
    state_mat['pitch'].append(eulers[1])
    state_mat['yaw'].append(eulers[2])

    location = location/100
    state_mat['x'].append(location[0])
    state_mat['y'].append(location[1])
    state_mat['z'].append(location[2])

    imu = imu
    print (imu[0])
    print (imu[1])
    print (imu[2])
    state_mat['p'].append(imu[0].copy())
    state_mat['q'].append(imu[1].copy())
    state_mat['r'].append(imu[2].copy())

    velocity = velocity/100
    state_mat['u'].append(velocity[0])
    state_mat['v'].append(velocity[1])
    state_mat['w'].append(velocity[2])



# state_mat = {'p':[],'q':[],'r':[]}
state_mat = {'roll_c':[],'roll':[],'pitch_c':[],'pitch':[],'yaw_c':[],'yaw':[],'alt_c':[],'alt':[],'x':[],'y':[],'z':[],
             'p':[],'q':[],'r':[],'u':[],'v':[],'w':[]}
roll = 0
pitch = 0
yaw = 0
altitude = 0
ALTITUDE_UP = K_w
ALTITUDE_DOWN = K_s
YAW_CCW = K_a
YAW_CW = K_d

PITCH_FWD = K_UP
PITCH_BACK = K_DOWN
ROLL_RIGHT = K_RIGHT
ROLL_LEFT = K_LEFT

QUIT = K_ESCAPE
firstframe=True
while running:

    pygame.event.pump() #this clears events to make sure fresh ones come in
    keys = pygame.key.get_pressed() #this looks for keys that are pressed

    #Mixer that sends keys pressed to commands
    if keys[ALTITUDE_UP]:
        altitude += .5
    if keys[ALTITUDE_DOWN]:
        if altitude >0:
            altitude -= .5
        else:
            altitude =0
    if keys[YAW_CW]:
        yaw = -2
    elif keys[YAW_CCW]:
        yaw = 2
    else:
        yaw = 0
    if keys[PITCH_FWD]:
        pitch = -.5
    elif keys[PITCH_BACK]:
        pitch = .5
    else:
        pitch = 0
    if keys[ROLL_LEFT]:
        roll = .5
    elif keys[ROLL_RIGHT]:
        roll = -.5
    else:
        roll = 0

    command = np.array([roll,pitch,yaw,altitude])
    state = uav_teleop(command)
    #state extracts the sensor information
    pixels = state[Sensors.PRIMARY_PLAYER_CAMERA]
    orientation = state[Sensors.ORIENTATION_SENSOR]
    location = state[Sensors.LOCATION_SENSOR]
    velocity = state[Sensors.VELOCITY_SENSOR]
    imu = state[Sensors.IMU_SENSOR]

    eulers = transforms3d.euler.mat2euler(orientation,'rxyz')
    update_state_mat(command,eulers,location,velocity,imu)

    # #the next block is looks for slider positions to determine filters
    frame = pixels
    if firstframe == True:
        src = frame
        firstframe=False

    cv2.imshow('image',src)
    # get current positions of four trackbars
    if cv2.getTrackbarPos('ksize H','image') %2 == 1:
        ksizeH = cv2.getTrackbarPos('ksize H','image')
    else:
        ksizeH = cv2.getTrackbarPos('ksize H','image')+1
    if cv2.getTrackbarPos('ksize H','image') <=0:
        ksizeH = 1;
    if cv2.getTrackbarPos('ksize V','image') %2 == 1:
        ksizeV = cv2.getTrackbarPos('ksize V','image')
    else:
        ksizeV = cv2.getTrackbarPos('ksize V','image')+1
    if cv2.getTrackbarPos('ksize V','image') <=0:
        ksizeV = 1;
    minval = cv2.getTrackbarPos('minval','image')
    maxval = cv2.getTrackbarPos('maxval','image')
    s0 = cv2.getTrackbarPos(switch0,'image')
    s = cv2.getTrackbarPos(switch,'image')
    s2 = cv2.getTrackbarPos(switch2,'image')
    s3 = cv2.getTrackbarPos(switch3,'image')
    ksize = (ksizeH,ksizeV)

    dia = cv2.getTrackbarPos('diameter','image')
    sigCol = cv2.getTrackbarPos('sigmaColor','image')
    sigSpace = cv2.getTrackbarPos('sigmaSpace','image')

    if s0 ==1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bilsrc = frame
    if s == 0 and s2 ==0 and s3 == 0:
        src = frame
        edgesrc = frame
        bilsrc = frame
    elif s == 0 and s2 == 1 and s3 == 0:
        src = edges
        edgesrc = frame
        bilsrc = frame
    elif s == 1 and s2 == 0 and s3 == 0:
        src = blur
        edgesrc = frame
        bilsrc = frame
    elif s == 1 and s2 == 1 and s3 == 0:
        src = edges
        edgesrc = blur
        bilsrc = frame
    elif s == 0 and s2 == 0 and s3 == 1:
        src = bilblur
        edgesrc = frame
        bilsrc = frame

    elif s == 0 and s2 == 1 and s3 == 1:
        src = edges
        edgesrc = bilblur
        bilsrc = frame
    else:
        src = edges
        edgesrc = blur
        bilsrc = frame

    blursrc = frame
    sigmaX = 0

    blur = cv2.GaussianBlur(blursrc,ksize,sigmaX)#[,dst[,sigmaY[,borderType]]])
    # bilblur = cv2.bilateralFilter(bilsrc,dia,sigCol,sigSpace)
    edges = cv2.Canny(edgesrc, minval, maxval)

    if savefile == True:
        sio.savemat(outfile,state_mat)


    if keys[QUIT]:
        running = False
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows
