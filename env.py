import numpy as np
from scipy.spatial import distance
from scipy.optimize import fsolve
from sympy import sympify,solve
from kinematics import Kinematics
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#from skimage import measure
from bencatel import Allen_Ben

class Node:
    def __init__(self,x,y,z):
        # x,y,z : State
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
class Edge:
    def __init__(self,cr,fpa,duration):
        # cr,fpa : control input
        self.cr = cr
        self.fpa = fpa
        # duration  is a coasting time
        self.dur = duration
        # edges are the curves between start_node and end_node
        self.sta = None
        self.end = None
        
# set goal and start
class Tree:
    def __init__(self,goal,start):
        self.start = start
        self.goal = goal
        
        self.r = 4.0
        self.V = set()
        self.E = set()
        
        # QE,QV
        self.QE = set()
        self.QV = set()
        
        # Parent Nodes
        self.v_old = set()
        
''' Thermal and obstacles setting '''
class Obstacles:
    def __init__(self,xyz0,abc,shape):
        '''
        x0,y0,z0 : obstacle center
        a,b,c : axes length of the obstacle
        d,e,f : shape parameters 
        -> 
        If d = e = 1 ,f > 1 : a cylinder
           d > 1 , e > 1, f > 1 : a cuboid 
        '''
        self.xyz0 = xyz0
        self.abc = abc  
        self.shape = shape 
        self.F = None
    def map(self,pos):
        x,y,z = pos[0],pos[1],pos[2]
        base = np.array([(x - self.xyz0[0])/self.abc[0],y - self.xyz0[1]/self.abc[1],z - self.xyz0[2]/self.abc[2]])
        exponenet = np.array(2 * self.shape[0], 2 * self.shape[1], 2 * self.shape[2])
        self.F = np.sum(np.power(base,exponenet)) - 1
        return self.F
    
    def draw(self):
        ''' Drawing the obstacles' edges '''
        # 음함수 방정식 정의
        def f(x,y,z,xyz0,abc,shape):
            x0, y0, z0 = xyz0
            a, b, c = abc
            d, e, f = shape
            return ((x-x0)/a)**(2*d) + ((y-y0)/b)**(2*e) + ((z-z0)/c)**(2*f) - 1

        # 플롯 범위 설정
        x_max, x_min = self.xyz0[0] + self.abc[0], self.xyz0[0] - self.abc[0]
        y_max, y_min = self.xyz0[1] + self.abc[1], self.xyz0[1] - self.abc[1]
        z_max, z_min = self.xyz0[2] + self.abc[2], self.xyz0[2] - self.abc[2]
        gap = 5
        interval = [x_min - gap, x_max + gap, y_min - gap, y_max + gap, z_min - gap, z_max + gap]
        
        # 3D 좌표 격자 생성 (20개)
        x, y, z = np.mgrid[interval[0]:interval[1]:20j, interval[2]:interval[3]:20j, interval[4]:interval[5]:20j]

        # Marching Cubes 알고리즘으로 곡면 추출
        verts, faces, normals, values = measure.marching_cubes(f(x, y, z, self.xyz0, self.abc, self.shape), 0)

        # 곡면 중심 계산
        center = (np.max(verts, axis=0) + np.min(verts, axis=0)) / 2

        # 곡면 데이터 평행 이동 
        verts -= center           # mesh에 맞춰 우선 원점으로 정렬
        verts += self.xyz0        # obstacle center에 맞게 평행이동

        # 3D 플롯 생성
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # 삼각형 메쉬 플롯
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, cmap='viridis', alpha=0.8, edgecolor='black',)

        # 축 설정 및 표시
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect((np.ptp(interval[0:2]), np.ptp(interval[2:4]), np.ptp(interval[4:6])))  # 축 비율 조정(axis equal)
        plt.show()
        
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
        if type == "chimmney":
            ABen = Allen_Ben()
            z = 800 # Our wanted estimation.
            X, Y, R, w_total = ABen.estimate(z)
        elif type == "bubble":
            pass
        elif type == "plume":
            pass
        elif type == "elongated":
            pass
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
# To keep the translational invariance, WIND vector의 discretize로 정리 ->  the database of motion primitives.

# Cost model -> bit.py에 있음
'''
def cost(state, control_input,duration):
    # state : [x,y,z]
    # control_input : [course rate, fpa]
    # duration : t_d
    g = 9.81
    kp = g * 
    T = max(D + m*g*np.sin(fpa),0)
    cost_f =  T * kp # internal fuel consumption
    cost_d = c(x,xt,u1,u2) / va
    
    print(cost_h,cost_t,cost_f)
    return cost_h + cost_t + cost_f

# Motion Primitives
'''
