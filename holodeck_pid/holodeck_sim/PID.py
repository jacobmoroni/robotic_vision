import math
import numpy as np

class PID:
    def __init__(self, kp, kd, ki, u_min=-math.inf, u_max=math.inf, angle_wrap=False):
        self.kp = kp
        self.kd = kd
        self.ki = ki

        self.min = u_min
        self.max = u_max

        self.integrator = 0.0
        self.error_prev = 0.0

        self.angle_wrap = angle_wrap

    def compute_control(self, x, x_c, dt):
        # Get the error
        error = x_c - x

        # Handle angle wrap, if desired
        if self.angle_wrap:
            if error > math.pi:
                error -= 2*math.pi
            elif error < -math.pi:
                error += 2*math.pi

        # Compute each component
        P = self.kp*error
        I = self.integrator + self.ki*error*dt
        D = self.kd*(error - self.error_prev)/dt

        # Compute control output
        u = P + I + D
        u_sat = self.saturate(u, self.min, self.max)

        # Update persistent values
        if u == u_sat:
            self.integrator = I # Help prevent integrator windup
        self.error_prev = error

        return u_sat

    def saturate(self, x, min_val, max_val):
        return max(min(x, max_val),min_val)

    def reset(self):
        # Reinitialize controller with same parameters
        self.__init__(self.kp, self.kd, self.ki, self.min, self.max, self.angle_wrap)
