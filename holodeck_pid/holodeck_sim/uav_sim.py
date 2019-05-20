import numpy as np
import math
import pygame
import transforms3d
from pygame.locals import *

from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
# from uav_state_plotter import Plotter
from state_plotter import Plotter
from PID import PID

### Command key mappings ###
# Basic commands
ROLL_RIGHT  = K_d
ROLL_LEFT   = K_a
PITCH_UP    = K_w
PITCH_DOWN  = K_s
YAW_LEFT    = K_LEFT
YAW_RIGHT   = K_RIGHT
ALT_UP      = K_UP
ALT_DOWN    = K_DOWN
SPEED_UP    = K_e
SPEED_DOWN  = K_q
# Velocity commands
VEL_FORWARD = K_w
VEL_BACKWARD= K_s
VEL_RIGHT   = K_d
VEL_LEFT    = K_a
# System commands
QUIT        = K_ESCAPE
RESET       = K_HOME
PAUSE       = K_SPACE
MANUAL_TOGGLE = K_LCTRL


class UAVSim():
    def __init__(self, world):
        ### Parameters
        # Default command
        self.roll_c = 0.0
        self.pitch_c = 0.0
        self.yawrate_c = 0.0
        self.alt_c = 0.0
        self.yaw_c = 0.0
        self.command = np.array([self.roll_c, self.pitch_c, self.yawrate_c, self.alt_c])

        # Rate parameters
        self.roll_min = math.radians(10)
        self.roll_max = math.radians(45)
        self.pitch_min = math.radians(10)
        self.pitch_max = math.radians(45)
        self.yawrate_min = math.radians(30)
        self.yawrate_max = math.radians(360)
        self.altrate_min = 0.1
        self.altrate_max = 0.5
        self.speed_min = 0.0
        self.speed_max = 1.0
        self.speed_rate = 0.05
        self.speed_val = 0

        # Velocity parmeters
        self.velocity_teleop = True
        self.vx_c   = 0.0
        self.vy_c   = 0.0
        self.vy_min = 0.5
        self.vy_max = 4.0
        self.vx_min = 0.5
        self.vx_max = 4.0

        # PID controllers
        vx_kp = -0.3
        vx_kd = -0.05
        vx_ki = -0.01
        self.vx_pid = PID(vx_kp, vx_kd, vx_ki, u_min=-self.pitch_max, u_max=self.pitch_max)
        vy_kp = 0.3
        vy_kd = 0.05
        vy_ki = 0.01
        self.vy_pid = PID(vy_kp, vy_kd, vy_ki, u_min=-self.roll_max, u_max=self.roll_max)
        yaw_kp = 5.0
        yaw_kd = 2.5
        yaw_ki = 0.00
        self.yaw_pid = PID(yaw_kp, yaw_kd, yaw_ki, u_min=-self.yawrate_max, u_max=self.yawrate_max, angle_wrap=True)

        # Teleop
        self.using_teleop = False
        self.teleop_text = "Click here to use teleop"

        # Simulation return variables
        self.sim_state = 0
        self.sim_reward = 0
        self.sim_terminal = 0
        self.sim_info = 0
        self.sim_step = 0
        self.dt = 1.0/30 # 30 Hz

        # Sensor data
        self.camera_sensor = np.zeros((512,512,4))
        self.position_sensor = np.zeros((3,1))
        self.orientation_sensor = np.identity(3)
        self.imu_sensor = np.zeros((6,1))
        self.velocity_sensor = np.zeros((3,1))

        # Default system variables
        self.plotting_states    = False
        self.paused             = False
        self.manual_control     = True

        # Initialize world
        print("Initializing {0} world".format(world))
        self.env = Holodeck.make(world)
        self.pressed = {PAUSE: False, MANUAL_TOGGLE: False}


    ######## Plotting Functions ########
    def init_plots(self, plotting_freq):
        self.plotting_states = True
        self.plotter = Plotter(plotting_freq)
        # Define plot names
        plots = ['x',                   'y',                    ['z', 'z_c'],
                 ['xdot', 'xdot_c'],    ['ydot', 'ydot_c'],     'zdot',
                 ['phi', 'phi_c'],      ['theta', 'theta_c'],   ['psi', 'psi_c'],
                 'p',                   'q',                    ['r', 'r_c'],
                 'ax',                  'ay',                   'az'
                 ]
        # Add plots to the window
        for p in plots:
            self.plotter.add_plot(p)

        # Define state vectors for simpler input
        self.plotter.define_state_vector("position", ['x', 'y', 'z'])
        self.plotter.define_state_vector("velocity", ['xdot', 'ydot', 'zdot'])
        self.plotter.define_state_vector("orientation", ['phi', 'theta', 'psi'])
        self.plotter.define_state_vector("imu", ['ax', 'ay', 'az', 'p', 'q', 'r'])
        self.plotter.define_state_vector("command", ['phi_c', 'theta_c', 'r_c', 'z_c'])
        self.plotter.define_state_vector("vel_command", ['xdot_c', 'ydot_c', 'psi_c'])


    ######## Teleop Functions ########
    def init_teleop(self):
        self.using_teleop = True
        pygame.init()
        SURFACE_WIDTH = 640
        SURFACE_HEIGHT = 200
        self.teleop_screen = pygame.display.set_mode( (SURFACE_WIDTH,SURFACE_HEIGHT) )
        pygame.display.set_caption('Holodeck UAV Teleop')

        self.teleop_font = pygame.font.Font(None, 50)

    def update_teleop_display(self):
        self.teleop_screen.fill((0,0,0))
        block = self.teleop_font.render(self.teleop_text, True, (255,255,255))
        rect = block.get_rect()
        rect.center = self.teleop_screen.get_rect().center
        self.teleop_screen.blit(block, rect)
        pygame.display.flip()

    def process_teleop(self):
        pygame.event.pump()
        keys=pygame.key.get_pressed()

        self.teleop_system_events(keys)
        if self.manual_control:
            self.teleop_commands(keys)


    def teleop_system_events(self,keys):
        # Update control values
        if keys[QUIT]:
            # return False # Quit the program
            self.exit_sim()

        if keys[RESET]:
            self.teleop_text = "Position reset"
            self.reset_sim()

        if keys[PAUSE]:
            # Only trigger on edge
            if not self.pressed[PAUSE]:
                self.pressed[PAUSE] = True
                self.paused = self.paused ^ True
                if self.paused == True:
                    self.teleop_text = "Simulation paused"
                else:
                    self.teleop_text = "Simulation resumed"
            # TODO: add debouncing timer
        elif self.pressed[PAUSE]: # Reset edge variable
            self.pressed[PAUSE] = False
        if self.paused:
            return

        if keys[MANUAL_TOGGLE]:
            # Only trigger on edge
            if not self.pressed[MANUAL_TOGGLE]:
                self.pressed[MANUAL_TOGGLE] = True
                self.manual_control = self.manual_control ^ True
                if self.manual_control == True:
                    self.teleop_text = "Manual control mode"
                else:
                    self.teleop_text = "Automatic control mode"
            # TODO: add debouncing timer
        elif self.pressed[MANUAL_TOGGLE]: # Reset edge variable
            self.pressed[MANUAL_TOGGLE] = False

    def teleop_commands(self, keys):
        # Default all angles/rates to zero at each time step
        if not self.velocity_teleop:
            self.roll_c = 0.0
            self.pitch_c = 0.0
            self.yawrate_c = 0
        else:
            self.vx_c = 0.0
            self.vy_c = 0.0

        # Lateral motion
        if self.velocity_teleop:
            if keys[VEL_RIGHT]:
                self.vy_c = (self.vy_min + (self.vy_max - self.vy_min)*self.speed_val)
                self.teleop_text = "VEL_RIGHT"
            if keys[VEL_LEFT]:
                self.vy_c = -(self.vy_min + (self.vy_max - self.vy_min)*self.speed_val)
                self.teleop_text = "VEL_LEFT"
            if keys[VEL_FORWARD]:
                self.vx_c = (self.vx_min + (self.vx_max - self.vx_min)*self.speed_val)
                self.teleop_text = "VEL_FORWARD"
            if keys[VEL_BACKWARD]:
                self.vx_c = -(self.vx_min + (self.vx_max - self.vx_min)*self.speed_val)
                self.teleop_text = "VEL_BACKWARD"
            # z-rotation
            if keys[YAW_LEFT]:
                self.yaw_c += (self.yawrate_min + (self.yawrate_max - self.yawrate_min)*self.speed_val)*self.dt
                if self.yaw_c > math.pi:
                    self.yaw_c -= 2*math.pi
                self.teleop_text = "YAW_LEFT"
            if keys[YAW_RIGHT]:
                self.yaw_c -= (self.yawrate_min + (self.yawrate_max - self.yawrate_min)*self.speed_val)*self.dt
                if self.yaw_c < -math.pi:
                    self.yaw_c += 2*math.pi
                self.teleop_text = "YAW_RIGHT"
        else:
            if keys[ROLL_RIGHT]:
                self.roll_c = (self.roll_min + (self.roll_max - self.roll_min)*self.speed_val)
                self.teleop_text = "ROLL_RIGHT"
            if keys[ROLL_LEFT]:
                self.roll_c = -(self.roll_min + (self.roll_max - self.roll_min)*self.speed_val)
                self.teleop_text = "ROLL_LEFT"
            if keys[PITCH_UP]:
                self.pitch_c = (self.pitch_min + (self.pitch_max - self.pitch_min)*self.speed_val)
                self.teleop_text = "PITCH_UP"
            if keys[PITCH_DOWN]:
                self.pitch_c = -(self.pitch_min + (self.pitch_max - self.pitch_min)*self.speed_val)
                self.teleop_text = "PITCH_DOWN"
            # z-rotation
            if keys[YAW_LEFT]:
                self.yawrate_c = (self.yawrate_min + (self.yawrate_max - self.yawrate_min)*self.speed_val)
                self.teleop_text = "YAW_LEFT"
            if keys[YAW_RIGHT]:
                self.yawrate_c = -(self.yawrate_min + (self.yawrate_max - self.yawrate_min)*self.speed_val)
                self.teleop_text = "YAW_RIGHT"
        # Altitude
        if keys[ALT_UP]:
            self.alt_c += (self.altrate_min + (self.altrate_max - self.altrate_min)*self.speed_val)
            self.teleop_text = "Altitude raised to {0:.1f}".format(self.alt_c)
        if keys[ALT_DOWN]:
            self.alt_c -= max(((self.altrate_min + (self.altrate_max - self.altrate_min)*self.speed_val), 0))
            self.teleop_text = "Altitude lowered to {0:.1f}".format(self.alt_c)
        # Speed/scaling
        if keys[SPEED_UP]:
            self.speed_val += self.speed_rate
            self.speed_val = min(self.speed_val, self.speed_max)
            self.teleop_text = "Speed raised to {0:.2f}".format(self.speed_val)
        if keys[SPEED_DOWN]:
            self.speed_val -= self.speed_rate
            self.speed_val = max(self.speed_val, self.speed_min)
            self.teleop_text = "Speed lowered to {0:.2f}".format(self.speed_val)

        if self.velocity_teleop:
            # Update roll and pitch commands based on velocity commands
            self.compute_velocity_control()

        self.set_command(self.roll_c, self.pitch_c, self.yawrate_c, self.alt_c)


    ######## External commands ########
    def command_velocity(self, vx, vy, yaw, alt):
        if self.manual_control:
            return

        self.vx_c = vx
        self.vy_c = vy
        self.yaw_c = yaw
        self.alt_c = alt

        self.compute_velocity_control()
        self.set_command(self.roll_c, self.pitch_c, self.yawrate_c, self.alt_c)


    ######## Control ########
    def compute_velocity_control(self):
        # Get current state
        vel = self.get_body_velocity()
        vx = vel[0]
        vy = vel[1]
        yaw = self.get_euler()[2]

        # Compute PID control
        self.pitch_c = self.vx_pid.compute_control(vx, self.vx_c, self.dt)
        self.roll_c = self.vy_pid.compute_control(vy, self.vy_c, self.dt)
        self.yawrate_c = self.yaw_pid.compute_control(yaw, self.yaw_c, self.dt)


    ######## Data access ########
    def set_command(self, roll, pitch, yawrate, alt):
        self.command = np.array([-roll, pitch, yawrate, alt]) # Roll command is backward in sim

    def extract_sensor_data(self):
        self.camera_sensor      = self.sim_state[Sensors.PRIMARY_PLAYER_CAMERA]
        self.position_sensor    = np.ravel(self.sim_state[Sensors.LOCATION_SENSOR])
        self.velocity_sensor    = np.ravel(self.sim_state[Sensors.VELOCITY_SENSOR])#/100.0 # Currently in cm - convert to m
        self.imu_sensor         = np.ravel(self.sim_state[Sensors.IMU_SENSOR])
        self.orientation_sensor = self.sim_state[Sensors.ORIENTATION_SENSOR]

    def get_state(self):
        return self.sim_state

    def get_camera(self):
        return self.camera_sensor

    def get_position(self):
        return self.position_sensor

    def get_world_velocity(self):
        return self.velocity_sensor

    def get_body_velocity(self):
        R = self.orientation_sensor
        world_vel = self.velocity_sensor
        body_vel = np.ravel(np.dot(R,world_vel)) # Rotate velocities into the body frame
        return body_vel

    def get_imu(self):
        self.imu_sensor[5] *= -1.0 # Yaw output seems to be backwards
        return self.imu_sensor

    def get_orientation(self):
        return self.orientation_sensor

    def get_euler(self):
        R = self.orientation_sensor
        euler = transforms3d.euler.mat2euler(R, 'rxyz')
        return euler

    def step_sim(self):
        if self.using_teleop:
            self.process_teleop()
            self.update_teleop_display()

        # Step holodeck simulator
        if not self.paused:
            self.sim_step += 1
            self.sim_state, self.sim_reward, self.sim_terminal, self.sim_info = self.env.step(self.command)
            self.extract_sensor_data() # Get and store sensor data from state
            if self.plotting_states:
                t = self.sim_step*self.dt
                self.plotter.add_vector_measurement("position", self.get_position(), t)
                self.plotter.add_vector_measurement("velocity", self.get_body_velocity(), t)
                self.plotter.add_vector_measurement("orientation", self.get_euler(), t)
                self.plotter.add_vector_measurement("imu", self.get_imu(), t)
                self.plotter.add_vector_measurement("command", self.command, t)
                self.plotter.add_vector_measurement("vel_command", [self.vx_c, self.vy_c, self.yaw_c], t)
                self.plotter.update_plots()

    def reset_sim(self):
        # Re-initialize commands
        self.set_command(0, 0, 0, 0)
        self.roll_c = 0.0
        self.pitch_c = 0.0
        self.yawrate_c = 0.0
        self.alt_c = 0.0
        self.vx_c = 0.0
        self.vy_c = 0.0
        self.yaw_c = 0.0

        # Re-initialize controllers
        self.vx_pid.reset()
        self.vy_pid.reset()
        self.yaw_pid.reset()

        # Reset the holodeck
        self.env.reset()

    def exit_sim(self):
        sys.exit()
