import pygame
from pygame.locals import *

DISPLAY_HEIGHT = 100
DISPLAY_WIDTH = 400

pygame.display.init()
pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
screen = DISPLAY_WIDTH, DISPLAY_HEIGHT
pygame.display.set_caption("Holodeck")
running = True
pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 30)
textsurface = myfont.render('Some Text', False, (0, 0, 0))
# screen.blit(textsurface,(0,0))
def holo(com):
    print (com)
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
print (altitude)
while running:

    pygame.event.pump()
    keys = pygame.key.get_pressed()
    if keys[ALTITUDE_UP]:
        altitude += .0001
    if keys[ALTITUDE_DOWN]:
        if altitude >0:
            altitude -= .0001
        else:
            altitude =0
    if keys[YAW_CW]:
        yaw = 1
    elif keys[YAW_CCW]:
        yaw = -1
    else:
        yaw = 0
    if keys[PITCH_FWD]:
        pitch = -1
    elif keys[PITCH_BACK]:
        pitch = 1
    else:
        pitch = 0
    if keys[ROLL_LEFT]:
        roll = -1
    elif keys[ROLL_RIGHT]:
        roll = 1
    else:
        roll = 0
    command = [roll,pitch,yaw,altitude]
    holo(command)
# while running:
#     clock.tick(fps_cap)
#
#     for event in pygame.event.get(): #error is here
#         if event.type == pygame.QUIT:
#             running = False
#
#     screen.fill(white)
#
#     pygame.display.flip()
#
# pygame.quit()
# sys.exit
#!/usr/bin/env python
