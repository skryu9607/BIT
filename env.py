import numpy as np
from scipy.spatial import distance
from scipy.optimize import fsolve
from sympy import sympify,solve
# set goal and start
class GS: # Goal and Start
    def __init__(self,goal,start):
        self.goal = goal
        self.start = start
    def shortest_length(self,a,b):
        return distance.euclidean(a,b)
''' Thermal and obstacles setting '''
class Obstacles:
    def __init__(self,x0,y0,z0,a,b,c,d,e,f):
        '''
        x0,y0,z0 : obstacle center
        a,b,c : axes length of the obstacle
        d,e,f : shape parameters 
        -> 
        If d = e = 1 ,f > 1 : a cylinder
           d > 1 , e > 1, f > 1 : a cuboid 
        '''
        self.xyz0 = np.array([x0,y0,z0])
        self.abc = np.array([a,b,c])  
        self.shape = np.array([d,e,f]) 
        self.F = None
    def map(self,pos):
        x,y,z = pos[0],pos[1],pos[2]
        base = np.array([(x - self.xyz0[0])/self.abc[0],y - self.xyz0[1]/self.abc[1],z - self.xyz0[2]/self.abc[2]])
        exponenet = np.array(2 * self.shape[0], 2 * self.shape[1], 2 * self.shape[2])
        self.F = np.sum(np.power(base,exponenet)) - 1
        return self.F
    def draw(self):
        ''' Drawing the obstacles' edges '''
        # F == 0인 line들만 합쳐서 그리기.
        expr = sympify(self.F)
        
    def collide(self,pos):
        if map(pos) < 0:
            return True
        else:
            return False

class Thermals:
    def __init__(self, location,type):
        self.type = type
        self.location = location
    def map(self):
        if type == "bubble":
            
        elif type == "plume":
            
        elif type == "elongated":
            
        else:
            print("This type is out of scope.")


# Wind set
class Winds: 
    def __init__(self,wind_intensity):
        self.speed = wind_intensity
    def wind_direction(self):
        wind_direct = []
        SET = [1,0,-1]
        for i in SET:
            for j in SET:
                for k in SET:
                     wind_direct.append([i,j,k])
        for i in range(27):
            if sum(wind_direct[i]) != 0:
                wind_direct[i] = [float(wind_direct[i][j])/sum(wind_direct[i]) for j in range(3)]
        return np.array(wind_direct)

# Cost model
class Cost:
    def __init__(self,location,wind)
# Motion Primitives
    

'''
Lattice에 random하게 winds가 분포된다. 세기도 random하게.


'''
class Map:
    det __init__

# Envs end


