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
    self.u_c = -command_input[0]
    self.v_c = -command_input[1]
    self.yaw_c = command_input[2]
    phi_c = -k1*(self.v_c-v)
    yaw_diff = self.yaw_c-yaw
    self.yaw_c = unwrap(self.yaw_c)
    yaw_diff = unwrap(yaw_diff)
    yaw_rate_c = k3*(yaw_diff)
    theta_c = -k2*(self.u_c-u)
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
    self.state_mat['u_c'].append(-command_input[0])
    self.state_mat['v_c'].append(-command_input[1])
    self.state_mat['yaw_c'].append(command_input[2])
    self.state_mat['roll_c'].append(command_output[0])
    self.state_mat['pitch_c'].append(command_output[1])
    self.state_mat['yaw_rate_c'].append(command_output[2])
    self.state_mat['alt_c'].append(command_output[3])

    self.state_mat['roll'].append(eulers[0])
    self.state_mat['pitch'].append(eulers[1])
    self.state_mat['yaw'].append(eulers[2])

    location = location/100
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
        self.atltitude_c += .5
    if keys[self.ALTITUDE_DOWN]:
        if self.atltitude_c >0:
            self.atltitude_c -= .5
        else:
            self.atltitude_c =0
    else:
        self.atltitude_c = self.atltitude_c
    if keys[self.YAW_CW]:
        self.yaw_c -= .05
    elif keys[self.YAW_CCW]:
        self.yaw_c += .05
    else:
        self.yaw_c = self.yaw_c
    if keys[self.FWD_VEL]:
        self.u_c = -5
    elif keys[self.BACK_VEL]:
        self.u_c = 5
    else:
        self.u_c = 0
    if keys[self.LEFT_VEL]:
        self.v_c = 5
    elif keys[self.RIGHT_VEL]:
        self.v_c = -5
    else:
        self.v_c = 0
    if keys[self.QUIT]:
        self.running = False
    return self.u_c,self.v_c,self.yaw_c,self.atltitude_c,self.running
