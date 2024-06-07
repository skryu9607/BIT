import numpy as np

class Kinematics:
    def __init__(self,x,y,z,heading,fpa):
        self.x = x
        self.y = y
        self.z = z
        self.fpa = fpa
        self.heading = heading
        self.dot_state = None
    def dot(self,wind,control_command,va):
        self.ctl = control_command
        dot_x = va * np.cos(self.fpa) * np.cos(self.heading) + wind[0]
        dot_y = va * np.cos(self.fpa) * np.sin(self.heading) + wind[1]
        dot_z = va * np.sin(self.fpa) + wind[2]
        dot_fpa = self.ctl[0]
        dot_heading = self.ctl[1]
        self.dot_state = [dot_x,dot_y,dot_z,dot_fpa,dot_heading].T
    def update(self):
        self.x += self.dot_state[0] * self.ctl[2]
        self.y += self.dot_state[1] * self.ctl[2]
        self.z += self.dot_state[2] * self.ctl[2]
        self.fpa += self.dot_state[3] * self.ctl[2]
        self.heading += self.dot_state[4] * self.ctl[2]

'''
It is assumed that an onboard controller is able to follow heading, airspeed, and throttle commands. 
Moreover, it is assumed that response to step changes in commands is very fast, compared with the duration of a particular command.
Hence, a point-mass model is sufficient to describe vehicle motion for planning purposes. 

'''
