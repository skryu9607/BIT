import numpy as np
from scipy.spatial import distance


# set goal and start
class GS: # Goal and Start
    def __init__(self,goal,start):
        self.goal = goal
        self.start = start
    def shortest_length(self,a,b):
        return distance.euclidean(a,b)
# Thermal and obstacles setting
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
    def map(self,pos):
        x,y,z = pos[0],pos[1],pos[2]
        base = np.array([(x - self.xyz0[0])/self.abc[0],y - self.xyz0[1]/self.abc[1],z - self.xyz0[2]/self.abc[2]])
        exponenet = np.array(2 * self.shape[0], 2 * self.shape[1], 2 * self.shape[2])
        F = np.sum(np.power(base,exponenet))
        return F
    def draw(self):
        
        
# Motion Primitives


# Wind set

# Lattice set

# Envs end

goal = [300,300,-2105]
start = [0,0,-1655]
GS(goal,start)
