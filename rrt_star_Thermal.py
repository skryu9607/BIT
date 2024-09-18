"""
Batch Informed Trees (BIT*) with thermal updrafts
@author : SeungKeol Ryu, Minjo Jung
"""

import os
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as Rot
import cProfile
import pstats
import io
import json
from winds import wind_catcher
#sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
#                "/../../Sampling_based_Planning/")

#from Sampling_based_Planning.rrt_2D import env, plotting, utils
from env import Node,Edge,Tree,Obstacles,Thermals
import utils
import plotting
from wind_model_v2 import WINDS, Thermals

# class BITStar => RRTStar
class RRTStar:
    def __init__(self, x_start, x_goal, eta, iter_max, va, u, v, w):
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.eta = eta
        self.iter_max = iter_max
        self.va = va
        self.delta = 10.0
        self.x_range = (-1000, 5000)
        self.y_range = (-1000, 5000)
        self.z_range = (0, 4000)
        
        self.fig = plt.figure(figsize=(15,12))
        self.ax = self.fig.add_subplot(111,projection = '3d')

        self.text = None
        
        '''
        Obstacles' shapes are assigned.
        '''

        self.Tree = Tree(self.x_start,self.x_goal)
        self.X_sample = set()
        # cost follow the tree
        # calculated by the cost accumulation followed by a series of parent node.
        self.g_T =  dict() 
        
        '''
        Wind Data
        '''
        self.u = u
        self.v = v
        self.w = w
        self.draw_things()
        
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
        xyz0 = [1000.,1000.,1000.]
        abc = np.array([200.,200.,200.])
        shape = np.array([1.,1.,2.])
        self.obs1 = Obstacles(xyz0,abc,shape,self.x_range,self.y_range,self.z_range)
        self.obs1.draw(self.ax)

    def prepare(self):
        self.Tree.V.add(self.x_start)
        self.X_sample.add(self.x_goal)
        
        self.g_T[self.x_start] = 0.0
        self.g_T[self.x_goal] = np.inf
        # At first glance, the batch size is just the distance between start point and the goal point. 
        distMin, _ = self.calc_dist_and_angle(self.x_start, self.x_goal)
        
        cMin = distMin/self.va
        
        # Center points
        xCenter = np.array([[(self.x_start.x + self.x_goal.x) / 2.0],
                            [(self.x_start.y + self.x_goal.y) / 2.0],
                            [(self.x_start.z + self.x_goal.z) / 2.0]])
        # Rotation matrix C
        self.C = self.RotationToWorldFrame(self.x_start, self.x_goal, distMin)
        print("Rotation Matrix is",self.C)
        return cMin,xCenter
    
    
    def planning(self):
        for i in range(self.iter_max):
            x_rand = self.sample()
            x_nearest = self.nearest(self.node_list, x_rand)
            x_new = self.steer(x_nearest, x_rand)
            
            if self.collision_free(x_nearest, x_new):
                neighbors = self.find_near_nodes(x_new)
                self.node_list.append(x_new)
                self.tree.add_node(x_new)
                self.tree.add_edge(Edge(x_nearest, x_new))
                x_min = x_nearest
                c_min = self.cost(x_nearest) + self.distance(x_nearest, x_new)
                
                for x_near in neighbors:
                    if self.collision_free(x_near, x_new) and self.cost(x_near) + self.distance(x_near, x_new) < c_min:
                        x_min = x_near
                        c_min = self.cost(x_near) + self.distance(x_near, x_new)
                        
                self.rewire(x_new, neighbors)
                
                if self.distance(x_new, self.x_goal) <= self.eta:
                    x_new_parent = self.nearest(self.node_list, self.x_goal)
                    if self.collision_free(x_new_parent, self.x_goal):
                        self.tree.add_edge(Edge(x_new_parent, self.x_goal))
                        self.x_goal.parent = x_new_parent
                        self.node_list.append(self.x_goal)
                        break
                    
        return self.extract_path()

    def sample(self):
        return Node((random.uniform(self.x_range[0], self.x_range[1]),
                     random.uniform(self.y_range[0], self.y_range[1]),
                     random.uniform(self.z_range[0], self.z_range[1])))

    def nearest(self, node_list, node):
        return min(node_list, key=lambda n: self.distance(n, node))

    def steer(self, from_node, to_node):
        distance = self.distance(from_node, to_node)
        if distance > self.eta:
            direction = ((to_node.x - from_node.x) / distance,
                         (to_node.y - from_node.y) / distance,
                         (to_node.z - from_node.z) / distance)
            new_node = Node((from_node.x + direction[0] * self.eta,
                             from_node.y + direction[1] * self.eta,
                             from_node.z + direction[2] * self.eta))
            new_node.parent = from_node
            return new_node
        return to_node

    def collision_free(self, node1, node2):
        # Check if the path between node1 and node2 is free of obstacles
        return True  # Placeholder for actual collision checking

    def distance(self, node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + 
                         (node1.y - node2.y) ** 2 + 
                         (node1.z - node2.z) ** 2)

    def cost(self, node):
        cost = 0.0
        while node.parent:
            cost += self.distance(node, node.parent)
            node = node.parent
        return cost

    def find_near_nodes(self, new_node):
        n_nodes = len(self.node_list) + 1
        r = self.eta * math.sqrt((math.log(n_nodes) / n_nodes))
        return [node for node in self.node_list if self.distance(node, new_node) <= r]

    def rewire(self, new_node, neighbors):
        for node in neighbors:
            if self.collision_free(new_node, node) and self.cost(new_node) + self.distance(new_node, node) < self.cost(node):
                node.parent = new_node
                self.tree.add_edge(Edge(new_node, node))

    def extract_path(self):
        path = []
        node = self.x_goal
        while node:
            path.append(node)
            node = node.parent
        return path[::-1]

    def main():
        # TODO : 
        x_start = (0.0, 0.0, 10.0)  # Starting node
        x_goal = (3000, 3000,3000)  # Goal node
        print("Start point is ", x_start)
        print("Goal point is ", x_goal)
        eta = 2 * 1 * 20 # radius 조절 parameter
        iter_max = 1000 
        va = 20 
        ResolutionType = 'normal'
        
        # Wind Data Path : 승걸
        # onedrive_path = '/Users/seung/WindData/'
        # Wind Data Path : 민조
        onedrive_path = 'C:/Users/LiCS/Documents/MJ/KAIST/Paper/2025 ACC/Wind Energy/Code/windData/'
        
        #Mac : OneDrive
        '''
        u = np.load(f'{onedrive_path}/{ResolutionType}/u_{ResolutionType}.npy')
        v = np.load(f'{onedrive_path}/{ResolutionType}/v_{ResolutionType}.npy')
        w = np.load(f'{onedrive_path}/{ResolutionType}/w_{ResolutionType}.npy')
        '''
        
        #Windows
        u = np.load(f'{onedrive_path}u_{ResolutionType}.npy')
        v = np.load(f'{onedrive_path}v_{ResolutionType}.npy')
        w = np.load(f'{onedrive_path}w_{ResolutionType}.npy')

        #print(u.shape,v.shape,w.shape)

        rrt = RRTStar(x_start, x_goal, eta, iter_max, va, u, v, w)
        #bit.draw_things()
        rrt.planning()
        print("start!!!")
        #bit = BITStar(x_start, x_goal, eta, iter_max)
        #bit.animation("Batch Informed Trees (BIT*)")
    
    if __name__== '__main__':
        main()
