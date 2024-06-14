"""
Batch Informed Trees (BIT*) with thermal updrafts
@author : SeungKeol Ryu
"""

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
from wind_model_v2 import WINDS, Thermals
class Tree:
    def __init__(self, x_start, x_goal):
        self.x_start = x_start
        self.goal = x_goal

        self.r = 4.0
        self.V = set()
        self.E = set()
        # So far now, I don't know exactly what these are.
        self.QE = set()
        self.QV = set()
        # Parent nodes.
        self.V_old = set()


class BITStar:
    def __init__(self, x_start, x_goal, eta, iter_max,va):
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
        self.y_range = (-5000, 5000)
        self.z_range = (0, 3000)
        '''
        Obstacles' shapes are assigned.
        '''
        self.Tree = Tree(self.x_start,self.x_goal)
        self.X_sample = set()
        # cost follow the tree
        # calculated by the cost accumulation followed by a series of parent node.
        self.g_T =  dict() 
    ''' 
    def WIND(self):
        # Ambient wind
        ambient_wind_speed = [1,0,0]
        amb = WINDS(seed = 1, wind_speed = ambient_wind_speed)
        # Thermal
        thm1 = Thermals(zi = 1000, w_star = 0, xc = 3000, yc = 3000)
        
        # Concatenation of each elements of wind vector
    '''    
    def draw_things(self):
        # Adding obstacles
        xyz0 = [1000,1000,1000]
        abc = np.array([10,10,10])
        shape = np.array([1,1,2])
        obs1 = Obstacles(xyz0,abc,shape)
        obs1.draw()
        # Adding thermal updrafts
        thermal_location = [2000,2000,2000]
        thm1 = Thermals(thermal_location, type = "chimmney")
        thm1.draw()
        
    def prepare(self):
        self.Tree.V.add(self.x_start)
        self.X_sample.add(self.x_goal)
        
        self.g_T[self.x_start] = 0.0
        self.g_T[self.x_goal] = np.inf
        # At first glance, the batch size is just the distance between start point and the goal point. 
        cMin, theta = self.calc_dist_and_angle(self.x_start, self.x_goal)
        # Center points
        xCenter = np.array([[(self.x_start.x + self.x_goal.x) / 2.0],
                            [(self.x_start.y + self.x_goal.y) / 2.0],
                            [(self.x_start.z + self.x_goal.z) / 2.0]])
        # Rotation matrix C.
        C = self.RotationToWorldFrame(self.x_start, self.x_goal, Lmin)
        
        return cMin,theta,xCenter,C
    
    def planning(self):
        cMin,theta,xCenter, C = self.prepare()
        #if self.Tree.QV is None:
            #print()
        for k in range(2500):
            # Batch Creation
            if not self.Tree.QE and not self.Tree.QV:
                if k == 0:
                    m = 350 * 6
                else:
                    m = 200 * 2
                # Reach goal points
                if self.x_goal.parent is not None:
                    path_x, path_y = self.ExtractPath()
                    plt.plot(path_x, path_y, linewidth=2, color='r')
                    plt.pause(0.01)
                # g_T :  Current Tree 구조상에서의 cost-to
                # self.Prune(기준 cost.), 목적지까지 가는 cost-to-come보다 작은 vertices들은
               #은 삭제된다. 
                self.Prune(self.g_T[self.x_goal])
               # update : inserting. 
               # sample할 때의 sample ellipsoid의 크기가 달라진다. 
                print(m)
                print(cMin)
                self.X_sample.update(self.Sample(m, self.g_T[self.x_goal], cMin, xCenter, C))
                self.Tree.V_old = {v for v in self.Tree.V} 
                print(len(self.Tree.V))
                self.Tree.QV = {v for v in self.Tree.V}
                print(len(self.Tree.QV))
                if k == 0: 
                    self.Tree.r = 1.5
                else:
                    self.Tree.r = self.radius(len(self.Tree.V) + len(self.X_sample))
                # Printing cBest <- Infinity
                print("Expansion")
            # 확장이 benefit할 때까지.
            # Best Edge가 있는한, 그걸 확장해야한다. 
            while self.BestVertexQueueValue() <= self.BestEdgeQueueValue():
                print("The Length of QV ",len(self.Tree.QV))
                print("The Length of QE ",len(self.Tree.QE))
                self.ExpandVertex(self.BestInVertexQueue())
   
            # Best means "minimum". min(distance).
            vm, xm = self.BestInEdgeQueue()
            # BestInEdgeQueue : graph상에서의 진짜 g와 v,x사이의 직선거리, 
            # 그리고 h_estimated으로 목적지까지의 거리.
            self.Tree.QE.remove((vm, xm))
            
            if self.g_T[vm] + self.calc_dist(vm, xm) + self.h_estimated(xm) < self.g_T[self.x_goal]:
                actual_cost = self.cost(vm, xm)
                if self.g_estimated(vm) + actual_cost + self.h_estimated(xm) < self.g_T[self.x_goal]:
                    if self.g_T[vm] + actual_cost < self.g_T[xm]:
                        if xm in self.Tree.V:
                            # remove edges
                            edge_delete = set()
                            for v, x in self.Tree.E:
                                if x == xm:
                                    edge_delete.add((v, x))

                            for edge in edge_delete:
                                self.Tree.E.remove(edge)
                        else:
                            self.X_sample.remove(xm)
                            self.Tree.V.add(xm)
                            self.Tree.QV.add(xm)

                        self.g_T[xm] = self.g_T[vm] + actual_cost
                        self.Tree.E.add((vm, xm))
                        xm.parent = vm

                        set_delete = set()
                        for v, x in self.Tree.QE:
                            if x == xm and self.g_T[v] + self.calc_dist(v, xm) >= self.g_T[xm]:
                                set_delete.add((v, x))

                        for edge in set_delete:
                            self.Tree.QE.remove(edge)
            else:
                self.Tree.QE = set()
                self.Tree.QV = set()
            #print("k is", k)
            if k % 20 == 0:
                self.animation(xCenter, self.g_T[self.x_goal], cMin, theta)

        path_x, path_y, path_z = self.ExtractPath()
        plt.plot(path_x, path_y, path_z, linewidth=2, color='r')
        plt.pause(0.01)
        plt.show()

    def ExtractPath(self):
        node = self.x_goal
        path_x, path_y, path_z = [node.x], [node.y] ,[node.z]

        while node.parent:
            node = node.parent
            path_x.append(node.x)
            path_y.append(node.y)
            path_z.append(node.z)
        return path_x, path_y, path_z

    def Prune(self, cBest):
        self.X_sample = {x for x in self.X_sample if self.f_estimated(x) < cBest}
        self.Tree.V = {v for v in self.Tree.V if self.f_estimated(v) <= cBest}
        self.Tree.E = {(v, w) for v, w in self.Tree.E
                       if self.f_estimated(v) <= cBest and self.f_estimated(w) <= cBest}
        self.X_sample.update({v for v in self.Tree.V if self.g_T[v] == np.inf})
        self.Tree.V = {v for v in self.Tree.V if self.g_T[v] < np.inf}
    def wind(self,points):
        return [0,0,0]
    def interpolate_points(start, end, num_points):
        points = []
        # num_points at least two
        for i in range(num_points + 1):
            t = i / num_points
            x = (1 - t) * start.x + t * end.x
            y = (1 - t) * start.y + t * end.y
            z = (1 - t) * start.z + t * end.z
            points.append((x, y, z))
            
        return points
    
    def getting_tangential(self, pos, displacement_dir):
        wind_dir = self.normalize(self.wind(pos))
        V_dir = displacement_dir - wind_dir
        
        return self.va * V_dir * displacement_dir + np.linalg.norm(self.wind(pos)) * wind_dir * displacement_dir
        
    def cost(self, start, end):
        L0 = self.calc_dist(start,end)
        N = 1000
        Cost = 0
        PNTs = self.interpolate_points(start,end,N)
        l0 = self.normalize(end-start)
        # 모든 점들이 false라면,
        
        for i in range(N):
            if Obstacles.collide.is_collision(PNTs[i]):
                print("Collision, We will prune it.")
                return np.inf
            Velocity_Tan = self.getting_tangential(PNTs[i,:],l0)
            Cost += L0/N / (Velocity_Tan)
            
        return Cost
    
    # estimation part should be changed. Not only distance but also energy cost.
    def interpolate_points(point1, point2, num_points):

        x_values = np.linspace(point1[0], point2[0], num_points)
        y_values = np.linspace(point1[1], point2[1], num_points)
        z_values = np.linspace(point1[2], point2[2], num_points)
        return np.vstack((x_values, y_values,z_values)).T
    
    def heuristics(self,start,end, n = 5):
        interploation = self.interpolate_points(start,end,n)
        # Split하고 w_max를 더하자. 
        PNTs = self.interpolate_points(start,end,n)
        # PNTs 중에서 하나가 collide될 수도 있다. 
        worst_case = np.min(self.getting_tangential(PNTs[i],self.normalize(end-start)) for i in range(n) if not Obstacles.collide(PNTs[i]))
        return self.calc_dist(start,end)/worst_case
    
    def f_estimated(self, node):
        return self.g_estimated(node) + self.h_estimated(node)

    def g_estimated(self, node):
        #return self.calc_dist(self.x_start, node)
        return self.heurstics(self.x_start,node)
    def h_estimated(self, node):
        #return self.calc_dist(node, self.x_goal)
        return self.heurstics(node,self.x_goal)
    
    def Sample(self, m, cMax, cMin, xCenter, C):
        if cMax < np.inf:
            return self.SampleEllipsoid(m, cMax, cMin, xCenter, C)
        else:
            return self.SampleFreeSpace(m)

    def SampleEllipsoid(self, m, cMax, cMin, xCenter, C):
        r = [cMax / 2.0,
             math.sqrt(cMax ** 2 - cMin ** 2) / 2.0,
             math.sqrt(cMax ** 2 - cMin ** 2) / 2.0]
        L = np.diag(r)

        ind = 0
        delta = self.delta
        Sample = set()

        while ind < m:
            xBall = self.SampleUnitNBall()
            x_rand = np.dot(np.dot(C, L), xBall) + xCenter
            # node = Node(x_rand[(0, 0)], x_rand[(1, 0)])
            node = Node(x_rand[(0, 0, 0)], x_rand[(1, 0, 0)])
            in_obs = Obstacles.collide(node)
            in_x_range = self.x_range[0] + delta <= node.x <= self.x_range[1] - delta
            in_y_range = self.y_range[0] + delta <= node.y <= self.y_range[1] - delta
            in_z_range = self.z_range[0] + delta <= node.z <= self.z_range[1] - delta
            if not in_obs and in_x_range and in_y_range and in_z_range:
                Sample.add(node)
                ind += 1

        return Sample

    def SampleFreeSpace(self, m):
        delta = self.utils.delta
        Sample = set()

        ind = 0
        while ind < m:
            node = Node(random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                        random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
                        random.uniform(self.z_range[0] + delta, self.z_range[1] - delta))
            if Obstacles.collide(node):
                continue
            else:
                Sample.add(node)
                ind += 1

        return Sample

    def radius(self, q):
        cBest = self.g_T[self.x_goal]
        print(cBest)
        lambda_X = len([1 for v in self.Tree.V if self.f_estimated(v) <= cBest])
        #radius = 
        radius = 2 * self.eta * (1.5 * lambda_X / math.pi * math.log(q) / q) ** 0.5
        print("radius is ",radius)
        return radius

    def ExpandVertex(self, v):
        # QV에 있는 거 지우면서 EXPAND 시작하기.
        self.Tree.QV.remove(v) 
        # X_near은 Sample들 중에서 radius r 안에 들어오는 모든 X_samples. 
        print("X_sample's length",len(self.X_sample))
        X_near = {x for x in self.X_sample if self.calc_dist(x, v) <= self.Tree.r}
        print("X_near's length",len(X_near))
        for x in X_near:
            # estimated 는 어떻게 얻는걸까?
            # self.start로부터의 직선거리. <- g // self.goal까지의 거리 : h_estimated . 
            if self.g_estimated(v) + self.calc_dist(v, x) + self.h_estimated(x) < self.g_T[self.x_goal]:
                self.g_T[x] = np.inf
                self.Tree.QE.add((v, x))

        if v not in self.Tree.V_old:
            V_near = {w for w in self.Tree.V if self.calc_dist(w, v) <= self.Tree.r}

            for w in V_near:
                if (v, w) not in self.Tree.E and \
                        self.g_estimated(v) + self.calc_dist(v, w) + self.h_estimated(w) < self.g_T[self.x_goal] and \
                        self.g_T[v] + self.calc_dist(v, w) < self.g_T[w]:
                    self.Tree.QE.add((v, w))
                    if w not in self.g_T:
                        self.g_T[w] = np.inf

    def BestVertexQueueValue(self):
        if not self.Tree.QV:
            return np.inf

        return min(self.g_T[v] + self.h_estimated(v) for v in self.Tree.QV)

    def BestEdgeQueueValue(self):
        if not self.Tree.QE:
            return np.inf

        return min(self.g_T[v] + self.calc_dist(v, x) + self.h_estimated(x)
                   for v, x in self.Tree.QE)

    def BestInVertexQueue(self):
        if not self.Tree.QV:
            print("QV is Empty!")
            return None
        # {key : value}
        v_value = {v: self.g_T[v] + self.h_estimated(v) for v in self.Tree.QV}

        return min(v_value, key=v_value.get)

    def BestInEdgeQueue(self):
        if not self.Tree.QE:
            print("QE is Empty!")
            return None

        e_value = {(v, x): self.g_T[v] + self.calc_dist(v, x) + self.h_estimated(x)
                   for v, x in self.Tree.QE}
 
        return min(e_value, key=e_value.get)

    @staticmethod
    def SampleUnitNBall():
        while True:
            x, y, z = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1,1)
            if x ** 2 + y ** 2 + z ** 2 < 1:
                return np.array([[x], [y], [z]])
            
            
    @staticmethod
    def normalize(v):
        return v/np.linalg.norm(v)
    @staticmethod
    def RotationToWorldFrame(x_start, x_goal, L):
        a1 = np.array([[(x_goal.x - x_start.x) / L],
                       [(x_goal.y - x_start.y) / L], 
                       [(x_goal.z - x_start.z) / L]])
        e1 = np.array([[1.0], [0.0], [0.0]])
        # @ : matrix multiplication using the "@" operator in Numpy.
        M = a1 @ e1.T
        # Eigen value, Eigen vector, 
        # To find the optimal rotation between two 3D vectors.
        U, _, V_T = np.linalg.svd(M, True, True)
        #C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T
        C = V_T @ U.T
        return C

    @staticmethod
    def calc_dist(start, end):
        return math.hypot(start.x - end.x, start.y - end.y, start.z - end.z)

    @staticmethod
    def calc_dist_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        dz = node_end.z - node_start.z
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def animation(self, xCenter, cMax, cMin, theta):
        plt.cla()
        #self.plot_grid("Batch Informed Trees (BIT*)")

        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        for v in self.X_sample:
            plt.plot(v.x, v.y, v.z, marker='.', color='lightgrey', markersize='2')

        if cMax < np.inf:
            self.draw_ellipse(xCenter, cMax, cMin, theta)

        for v, w in self.Tree.E:
            plt.plot([v.x, w.x], [v.y, w.y], [v.z, w.z],'-g')

        plt.pause(0.001)

    def plot_grid(self, name):
        for (ox, oy, w, h) in self.obs_boundary:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor ='black',
                    facecolor ='black',
                    fill = True
                )
            )

        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        plt.plot(self.x_start.x, self.x_start.y, "bs", linewidth=3)
        plt.plot(self.x_goal.x, self.x_goal.y, "rs", linewidth=3)

        plt.title(name)
        plt.axis("equal")

    @staticmethod
    def draw_ellipse(x_center, c_best, dist, theta):
        #TODO : Eplliipsoid 표현해줘.
        '''
        x = a sin(phi) * cos(tha)
        y = b sin(phi) * sin(tha)
        z = c cos(phi)
        '''
        a = math.sqrt(c_best ** 2 - dist ** 2) / 2.0
        b = c_best / 2.0
        c = c_best / 2.0
        angle = math.pi / 2.0 - theta
        cx = x_center[0]
        cy = x_center[1]
        cz = x_center[2]
        t = np.arange(0, 2 * math.pi + 0.1, 0.2)
        phi = np.arange(0, 2 * math.pi + 0.1, 0.2)
    
        x = [a * math.sin(iphi) * math.cos(it)  for it in t for iphi in phi]
        y = [b * math.sin(iphi) * math.sin(it)  for it in t for iphi in phi]
        z = [c * math.cos(iphi) for iphi in phi]
        rot = Rot.from_euler('z', -angle).as_matrix()[0:2, 0:2]
        fx = rot @ np.array([x, y, z])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        pz = np.array(fx[2, :] + cz).flatten()
        plt.plot(cx, cy, cz, marker='.', color='darkorange')
        plt.plot(px, py, pz, linestyle='--', color='darkorange', linewidth=2)



def main():
    # TODO : 
    x_start = (100, 100, 0)  # Starting node
    x_goal = (3000, 3000, 3000)  # Goal node
    eta = 2 * 1
    iter_max = 200
    BITStar.init()
    BITStar.draw_things()
    print("start!!!")
    bit = BITStar(x_start, x_goal, eta, iter_max)
    bit.animation("Batch Informed Trees (BIT*)")
    bit.planning()

if __name__=="main":
    main()
    