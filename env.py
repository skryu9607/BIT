import numpy as np
from scipy.spatial import distance
from scipy.optimize import fsolve
from sympy import sympify,solve
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#from skimage import measure
from bencatel import allen_model_with_bencatel
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Map:
    def __init__(self, obs=None) :
        self.obs = obs

class Node:
    def __init__(self,X):
        # x,y,z : State
        self.x = X[0]
        self.y = X[1]
        self.z = X[2]
        self.xyz = np.array([self.x,self.y,self.z])
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
        
        self.r = 4.0 * 2
        self.V = set()
        self.E = set()
        
        # QE,QV
        self.QE = set()
        self.QV = set()
        
        # Parent Nodes
        self.v_old = set()
        
''' Thermal and obstacles setting '''
class Obstacle:
    def __init__(self, xyz0, abc, shape):
        '''
        x0, y0, z0 : obstacle center
        a, b, c : axes length of the obstacle
        d, e, f : shape parameters 
        -> 
        If d = e = 1 ,f > 1 : a cylinder
            d > 1 , e > 1, f > 1 : a cuboid 
        '''
        self.xyz0 = xyz0
        self.abc = abc  
        self.shape = shape 
        self.F = None 
        
    def map(self, pos):
        x, y, z = pos[0], pos[1], pos[2]
        base = np.array([(x - self.xyz0[0]) / self.abc[0], (y - self.xyz0[1]) / self.abc[1], (z - self.xyz0[2]) / self.abc[2]])
        exponent = np.array([2 * self.shape[0], 2 * self.shape[1], 2 * self.shape[2]])
        self.F = np.sum(np.power(base, exponent)) - 1
        return self.F

    def draw(self, ax):
        ''' Drawing the obstacles' edges '''
        # 음함수 방정식 정의
        def f(x, y, z, xyz0, abc, shape):
            x0, y0, z0 = xyz0
            a, b, c = abc
            d, e, f = shape
            return ((x-x0)/a)**(2*d) + ((y-y0)/b)**(2*e) + ((z-z0)/c)**(2*f) - 1

        # 플롯 범위 설정
        x_max, x_min = self.xyz0[0] + self.abc[0], self.xyz0[0] - self.abc[0]
        y_max, y_min = self.xyz0[1] + self.abc[1], self.xyz0[1] - self.abc[1]
        z_max, z_min = self.xyz0[2] + self.abc[2], self.xyz0[2] - self.abc[2]
        '''
        x_max,x_min = self.x_range[1],self.x_range[0]
        y_max,y_min = self.y_range[1],self.y_range[0]
        z_max,z_min = self.z_range[1],self.z_range[0]
        '''
        gap = 5
        
        interval = [x_min - gap, x_max + gap, y_min - gap, y_max + gap, z_min - gap, z_max + gap]
        #print(interval)

        # 3D 좌표 격자 생성
        grid_resolution = 10  # 해상도 조정 변수
        x, y, z = np.mgrid[interval[0]:interval[1]:complex(0, grid_resolution),
                           interval[2]:interval[3]:complex(0, grid_resolution),
                           interval[4]:interval[5]:complex(0, grid_resolution)]

        # Marching Cubes 알고리즘으로 곡면 추출
        verts, faces, normals, values = measure.marching_cubes(f(x, y, z, self.xyz0, self.abc, self.shape), 0)

        # 곡면 중심 계산
        center = (np.max(verts, axis=0) + np.min(verts, axis=0)) / 2

        # 곡면 데이터 스케일링 및 평행이동
        verts -= center                                     # mesh에 맞춰 우선 원점으로 정렬
        verts  = verts * self.abc / grid_resolution         # 장애물의 크기(abc)를 반영한 스케일링 (장애물은 해상도 크기에 맞게 그려지므로)
        verts += self.xyz0                                  # obstacle center에 맞게 평행이동

        # 삼각형 메쉬 플롯
        mesh = Poly3DCollection(verts[faces], alpha=0.8, edgecolor='black', facecolor='green')
        ax.add_collection3d(mesh)
        
        # ax_obstacle.set_xlim([np.min(verts[:, 0]), np.max(verts[:, 0])])
        # ax_obstacle.set_ylim([np.min(verts[:, 1]), np.max(verts[:, 1])])
        # ax_obstacle.set_zlim([np.min(verts[:, 2]), np.max(verts[:, 2])])
        ax_obstacle.view_init(elev=10, azim=-85)
        ax_obstacle.set_xlabel('X')
        ax_obstacle.set_ylabel('Y')
        ax_obstacle.set_zlabel('Z')
        ax_obstacle.set_box_aspect((np.ptp(interval[0:2]), np.ptp(interval[2:4]), np.ptp(interval[4:6])))  # 축 비율 조정(axis equal)
        ax_obstacle.auto_scale_xyz([self.x_range[0],self.x_range[1]], [self.y_range[0],self.y_range[1]], [self.z_range[0],self.z_range[1]])
        ax_obstacle.plot([verts_at_z_max[0]], [verts_at_z_max[1]], [verts_at_z_max[2]], marker='o', color='red', markersize=4)
        ax_obstacle.plot(x_start.x, x_start.y,x_start.z, marker='s', color='blue', markersize=4)
        ax_obstacle.plot(x_goal.x,x_goal.y,x_goal.z, marker='x', color='blue', markersize=4)
        
    def collide(self,pos):
        if self.map(pos) < 0:
            return True
        else:
            return False

class Thermals:
    def __init__(self, location,type):
        self.type = type
        self.location = location
    def map(self):
        if type == "chimmney":
            ABen = allen_model_with_bencatel()
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
