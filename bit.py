import os
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as Rot

#sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
#                "/../../Sampling_based_Planning/")

#from Sampling_based_Planning.rrt_2D import env, plotting, utils
from env import Node,Edge,Tree,Obstacles,Thermals
import utils
import plotting
# Static method : cl



class BITStar:
    def __init__(self,x_start,x_goal,eta,va = 22, iter_max = 3000):
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.eta = eta
        self.iter_max = iter_max
        self.va = va
        ''' TODO : after writing, for visualization, I have to do this.
        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start,x_goal)
        self.utils = utils.Utils()
        
        self.fig,self.ax = plt.subplots()
        '''
        self.delta = self.utils.delta
        self.x_range = (-5000, 5000)
        self.y_range = (-3000, 3000)
        self.z_range = (0, 3000)
        '''
        Obstacles' shapes are assigned.
        '''
        self.Tree = Tree(self.x_start,self.x_goal)
        self.X_sample = set()
        # cost follow the tree
        # calculated by the cost accumulation followed by a series of parent node.
        self.g_T =  dict() 
        
        
    def draw_things(self):
        # Adding obstacles
        xyz0 = [1000,1000,1000]
        abc = np.array([10,10,10])
        shape = np.array([1,1,2])
        obs1 = Obstacles(xyz0,abc,shape)
        obs1.draw()
        # Adding thermal updrafts
        thermal_location = [2000,2000,2000]
        thm1 = Thermals(thermal_location,type = "chimmney")
        thm1.draw()
        
    def prepare(self):
        self.Tree.V.add(self.x_start)
        self.X_sample.add(self.x_goal)
        
        self.g_T[self.x_start] = 0.0
        self.g_T[self.x_goal] = np.inf
        # At first glance, the batch size is just the distance between start point and the goal point. 
        cMin, theta = self.calc_dist_and_angle(self.x_start, self.x_goal)
        C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        # Rotation matrix C.
        xCenter = np.array([[(self.x_start.x + self.x_goal.x) / 2.0],
                            [(self.x_start.y + self.x_goal.y) / 2.0],
                            [(self.x_start.z + self.x_goal.z) / 2.0]])
        return theta,cMin,xCenter,C
    
    def planning(self):
        theta, cMin, xCenter, C = self.prepare()
        for slack1 in range(self.iter_max):
            # End of batch and also batch creation
            # if the set is empty, the set() is false.
            # self.Tree.QE is empty -> self.Tree.QE == False -> if () and () <- (),() have to be trues.
            if not self.Tree.QE and not self.Tree.QV:
                # NO.Expanding samples 
                if slack1 == 0:
                    m = 500 * 2
                else:
                    m = 500
                if self.x_goal.parent is not None:
                    path_x,path_y,path_z = self.ExtractSolution()
                    print("We Found the solution!")
                    plt.plot(path_x,path_y,path_z,linewidth = 2, color = 'r')
                    plt.pause(0.01)
                
                # Prune
                
                # Generating new samples
                x_new = self.sample(m)
                
                
                
    def ExtractSolution(self):
        path = self.x_goal
        path_x,path_y,path_z = path.x,path.y,path.z
        while path.parent:
            path = path.parent
            path_x.append(path.x)
            path_y.append(path.y)
            path_z.append(path.z)
            
        return path_x,path_y,path_z
                
                
    # TODO : sample(m)
    
    # TODO : near(rho, x, X)

    # TODO : 
           
    def prune(self,c_sol):
        '''
        Inputs : self.Tree, X_ncon (non connection), X_flags = []
        Outputs : X_reuse, self.Tree, X_ncon, X_flags= [] 
        '''
        X_reuse = set()
        X_ncon = {x for x in self.X_ncon if self.f_estimated(x)>=c_sol}
        self.Tree.V = {v for v in self.Tree.v if self.g_T[v] + self.h_estimated[v] <= c_sol}
        self.Tree.E = {(v,w,u1,u2) for v,w,u1,u2 in self.Tree.E if self.g_T[v]+self.h_estimated[v]<= c_sol and self.f_}
        for x in to_remove:
            self.Tree.v.discard(x)
            self.Tree.Q.discard(x.parent,x)
            
            
    def str_dist(start,end):
        return math.hypot(start.x-end.x,start.y-end.y,start.z-end.z)
    
    def cal_dist(start,end,u1,u2,duration):
        # TODO : the real cost value is different by not only obstacles but curves.
        
        pass
    def cost(self,x,u1,u2,vg,duration):
        ''' vg 이런거는 관계식이 아직 안 밝혀짐.'''
        D = 100 # 임의로 설정.
        m = 14 # 임의로 설정.
        g = 9.81
        T = max(D+m*g*np.sin(u2),0)
        fuel_const = 0.8
        fuel_consumption = T*self.va/(m* g *fuel_const)
        '''                Travel Time term              +     fuel term     '''
        return self.cal_dist(self.start,x,u1,u2,duration)/vg + (fuel_consumption)/vg
    
    def g_estimated(self,x):
        return 0 + self.str_dist(self.x_start, x) / (self.va)
    
    def h_estimated(self,x):
        return 0 + self.str_dist(x,self.x_goal) / (self.va)
    
    def f_estimated(self,x):
        return self.g_estimated(x) + self.h_estimated(x) / (self.va)
    
    def BestVertexQueueValue(self):
        if not self.Tree.QV:
            return np.inf

        return min(self.g_T[v] + self.h_estimated(v) for v in self.Tree.QV)
    
    def BestEdgeQueueValue(self):
        if not self.Tree.QE:
            return np.inf

        return min(self.g_T[v] + self.cal_dist(v, x, u1,u2) + self.h_estimated(x)
                   for v, x ,u1,u2 in self.Tree.QE)
        
    def BestInVertexQueue(self):
        if not self.Tree.QV:
            print("QV is empty")
            return None
        v_value = {v:self.g_T[v] + self.h_estimated[v] for v in self.QV}
        return min(v_value,key =v_value.get)
    
    def BestInEdgeQueue(self):
        if not self.Tree.QE:
            print("QE is empty")
            return None
        e_value = {(v,x,u1,u2) :}
        
        
    # static method는 정적메소드로, python instance 상태와 무관할때 유용하게 쓰인다.
    @staticmethod 
    def draw_ellipse(x_center, c_best, dist, theta):
        a = math.sqrt(c_best ** 2 - dist ** 2) / 2.0
        b = c_best / 2.0
        angle = math.pi / 2.0 - theta
        cx = x_center[0]
        cy = x_center[1]
        t = np.arange(0, 2 * math.pi + 0.1, 0.2)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        rot = Rot.from_euler('z', -angle).as_matrix()[0:2, 0:2]
        fx = rot @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, marker='.', color='darkorange')
        plt.plot(px, py, linestyle='--', color='darkorange', linewidth=2)        
        


def main():
    
    BITStar.draw_things()
    BITStar.init()



if __name__=="main":
    main()
    
