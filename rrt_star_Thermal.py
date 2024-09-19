"""
Rapidly-exploring Random Trees (RRT*) with thermal updrafts
@author : Minjo Jung, SeungKeol Ryu
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

class RRTStar:
    def __init__(self, x_start, x_goal, r, step_len, iter_max, va, u, v, w):
        
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        # self.eta = eta                # eta 제거
        self.gamma = step_len           # 한 스텝당 갈 수 있는 최대 거리 (step_len = gamma)
        self.iter_max = iter_max        # 최대 iteration
        self.va = va                    # 순항 속도?
        self.delta = 10.0               # 자유 공간 정의를 위한 안전 여유
        
        self.x_range = (-1000, 5000)    # 맵 범위 (X)
        self.y_range = (-1000, 5000)    # 맵 범위 (Y)
        self.z_range = (0, 4000)        # 맵 범위 (Z)
        
        self.Tree = Tree(self.x_start, self.x_goal)
        self.Tree.V.add(self.x_start)
        self.Tree.r = r                 # search radius
        
        self.g_T = dict()               # 각 노드 별 cost-to-come(g)
        self.g_T[self.x_start] = 0.0
        self.g_T[self.x_goal] = np.inf

        self.Flag = False               # Path Found 여부

        '''
        Visualization Setup
        '''
        self.fig = plt.figure(figsize=(15,12))
        self.ax = self.fig.add_subplot(111,projection = '3d')
        self.text = None

        '''
        Wind Data
        '''
        self.u = u
        self.v = v
        self.w = w

        '''
        Obstacles
        '''
        self.draw_things()
        
           
    '''Obstacle Def & Plot'''
    def draw_things(self):
        # Adding obstacles
        xyz0 = [1000.,1000.,1000.]
        abc = np.array([200.,200.,200.])
        shape = np.array([1.,1.,2.])
        self.obs1 = Obstacles(xyz0,abc,shape,self.x_range,self.y_range,self.z_range)
        self.obs1.draw(self.ax)

    ''' prepare 생략 : RRT*에서 쓰이지 않음 (Initialize)
    def prepare(self):
        self.Tree.V.add(self.x_start)
        self.X_sample.add(self.x_goal)
        
        self.g_T[self.x_start] = 0.0
        self.g_T[self.x_goal] = np.inf
        
        # At first glance, the batch size is just the distance between start point and the goal point. 
        distMin, _ = self.calc_dist_and_angle(self.x_start, self.x_goal)
        
        # 최소 cost [시간] = (직선 거리)/(va) 
        cMin = distMin/self.va
        
        # Center points
        xCenter = np.array([[(self.x_start.x + self.x_goal.x) / 2.0],
                            [(self.x_start.y + self.x_goal.y) / 2.0],
                            [(self.x_start.z + self.x_goal.z) / 2.0]])
       
        # Rotation matrix C
        self.C = self.RotationToWorldFrame(self.x_start, self.x_goal, distMin)
        print("Rotation Matrix is",self.C)
        
        return cMin, xCenter
    '''

    '''PATH PLANNING : RRT*'''
    def planning(self):
                 
        # Initialize for visualization
        self.ax.view_init(elev=60, azim=30)
        self.ax.scatter(self.x_start.x,self.x_start.y,self.x_start.z,marker = 's' ,color = 'blue',s = 20)
        self.ax.scatter(self.x_goal.x,self.x_goal.y,self.x_goal.z,marker = 'x' ,color = 'blue',s = 20)
        
        # Initialize for data storing
        file_path = "data.json"
        if os.path.exists(file_path):   # 기존 JSON 파일이 존재하는지 확인
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    # data_list = json.load(file)  # 기존 데이터를 리스트로 로드
                    data_list = []
                    if not isinstance(data_list, list):
                        data_list = []  # 만약 데이터가 리스트가 아니라면 빈 리스트로 초기화
                except json.JSONDecodeError:
                    data_list = []  # 파일이 비어있거나 잘못된 경우 빈 리스트로 초기화
        else:
            data_list = []

        # Main Planning (RRT*)
        path_x = []
        path_y = []
        path_z = []

        for k in range(self.iter_max):
            
            
            # 노드 랜덤 샘플링 (x_rand)
            x_rand = self.SampleRRT()

            # Nearest 노드 찾기 (x_nearest)
            x_nearest = self.nearest(x_rand)
            
            # New 노드 생성 (x_new) (step_len 고려해서) (충돌이 없으면, parent = x_nearest, 충돌이 있으면 parent = None)
            x_new = self.make_new(x_nearest, x_rand)
            
            if x_new.parent == None:                        # x_new.parent 가 None 이면, 다음 샘플링 루프로 넘김 (다시 샘플링)
                continue
            
            self.Tree.V.add(x_new)                          # Tree에 x_new 추가

            self.g_T[x_new] = self.g_estimated(x_nearest) + self.calc_dist(x_nearest, x_new)/self.va    # x_nearest를 통해 x_new 까지의 cost-to-come(g)

            # 1. parent 재선정
            near_nodes = self.find_near_nodes(x_new)
            for x_near in near_nodes:
                if self.collisionFree(x_near, x_new):   # x_near 부터 x_new 로의 경로 상에 충돌이 없는 경우 (feasible path 인 경우)
                    g_near = self.g_estimated(x_near) + self.calc_dist(x_near,x_new)/self.va
                    if g_near < self.g_T[x_new]:
                        x_new.parent = x_near
                        self.g_T[x_new] = g_near

            # 2. Rewiring (기존 노드들의 parent 재선정)
            self.rewire(x_new, near_nodes)
            
            # Tree Plot (매 iter 마다 plot 을 갱신하고 싶은데, 마지막에 한 번만 그려짐 ㅠㅠ)
            self.plot_tree()

            # x_new 노드가 goal 노드 일정 반경 내에 들어오고, x_goal까지 feasible 하면
            if self.calc_dist(x_new, self.x_goal) <= self.gamma and self.collisionFree(x_new, self.x_goal):
                self.x_goal.parent = x_new

            # Goal 에 도달한 경우, Path 추출 및 반복문 종료
            if self.x_goal.parent is not None:
                
                path_x, path_y, path_z = self.ExtractPath()     # Path 추출
                self.Flag = True                                # Success 여부
                break
        
        if not self.Flag:
            print(f"Solution Not Found within {self.iter_max} iteration.")
        else:
            print(f"Solution Found in {k}th iteration!")

        # Data Storing
        data = {
            "Start point": self.x_start.xyz.tolist(),  # 리스트 변환
            "Goal point": self.x_goal.xyz.tolist(),
            "Iteration number": k,
            "Path": [path_x, path_y, path_z],
            "Cost": self.g_T[self.x_goal],  # 목표점까지의 비용
        }
        
        data_list.append(data)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data_list, file, ensure_ascii=False, indent=4)

        print(f"데이터 저장 완료")

        # Plot
        plt.title("Wind Aware Batch Informed Trees")
        plt.xlabel("X")
        plt.ylabel("Y")
        # plt.zlabel("Z")
        plt.plot(path_x, path_y, path_z , linewidth=2, color='r',linestyle ='--')
        plt.show()

    ''' Tree 에서 node 와 가장 가까운 노드 (x_nearest) 찾기 '''
    def nearest(self, node):
        min_dist = 1000000000
        for v in self.Tree.V:
            dist = self.calc_dist(node, v)
            if dist < min_dist:
                min_dist = dist
                x_nearest = v
        
        return x_nearest
    
    ''' node 와 특정 반경 r 내에 위치한 가까운 노드들 반환 '''
    def find_near_nodes(self, node):
        near_nodes = set()
        for v in self.Tree.V:
            dist = self.calc_dist(node, v)
            if dist < self.Tree.r:
                near_nodes.add(v)
        return near_nodes
    
    ''' 새로운 노드 (x_new) 생성 '''
    def make_new(self, from_node, to_node):
        distance = self.calc_dist(from_node, to_node)
        if distance > self.gamma:
            direction = self.normalize(to_node.xyz - from_node.xyz)
            new_node = Node((from_node.x + direction[0] * self.gamma,
                             from_node.y + direction[1] * self.gamma,
                             from_node.z + direction[2] * self.gamma))
            if self.collisionFree(from_node, to_node):          # 경로 상에 장애물과 충돌이 없으면, 부모 노드 추가
                new_node.parent = from_node
            return new_node
        else:
            if self.collisionFree(from_node, to_node):
                to_node.parent = from_node
            return to_node

    ''' 두 노드 사이의 경로점들의 장애물 충돌 검사'''
    def collisionFree(self, from_node, to_node, n = 10):
        PNTs = self.interpolate_points(from_node,to_node,n)     # 경로의 각 점들
        for i in range(n):                                      # 모든 경로점들에 대해
            if self.obs1.collide(PNTs[i]):                      # 장애물과 충돌 검사 (해당 점이 장애물 내에 속하는지 아닌지)
                return False                                    # 하나라도 충돌하면 False (Collison free 하지 않다)
        return True                                             # 충돌이 없으면 True


    ''' 기존 이웃노드들의 parent 노드 갱신 '''
    def rewire(self, new_node, near_nodes):
        for node in near_nodes:
            if self.collisionFree(new_node, node) and self.g_estimated(new_node) + self.calc_dist(new_node, node)/self.va < self.g_estimated(node):
                node.parent = new_node
                self.g_T[node] = self.g_estimated(new_node) + self.calc_dist(new_node, node)/self.va

    '''path 의 x,y,z 추출'''
    def ExtractPath(self):
        node = self.x_goal
        path_x, path_y, path_z = [node.x], [node.y] ,[node.z]

        while node.parent:
            node = node.parent
            path_x.append(node.x)
            path_y.append(node.y)
            path_z.append(node.z)
        return path_x, path_y, path_z

    ''' Prune 생략 : RRT*에서 쓰이지 않음
    def Prune(self, cBest):
        self.X_sample = {x for x in self.X_sample if self.f_estimated(x) < cBest}
        self.Tree.V = {v for v in self.Tree.V if self.f_estimated(v) <= cBest}
        if len(self.Tree.V) == 0:
            print("The first stage of prune")
            print("The current cBest is ",cBest)
            for v in self.Tree.V:
                print(self.f_estimated(v))
        self.Tree.E = {(v, w) for v, w in self.Tree.E
                       if self.f_estimated(v) <= cBest and self.f_estimated(w) <= cBest}
        self.X_sample.update({v for v in self.Tree.V if self.g_T[v] == np.inf})
        self.Tree.V = {v for v in self.Tree.V if self.g_T[v] < np.inf}
        if len(self.Tree.V) == 0:
            print("The second stage of prune")
    '''

    '''해당 위치에서의 바람 성분 반환'''    
    def wind(self, points):
        #print(points)
        wind_at_node = wind_catcher(points[0],points[1],points[2],self.u,self.v,self.w)
        #print(wind)
        return wind_at_node
        #return [0.,0.,0.]

    ''' interpolate_points 생략 : 아래에 중복된 것이 이미 있음
    def interpolate_points(start, end, num_points):
        points = []
        # num_points at least two
        for i in range(num_points + 1):
            t = i / num_points
            x = (1 - t) * start.x + t * end.x
            y = (1 - t) * start.y + t * end.y
            z = (1 - t) * start.z + t * end.z
            points.append([x, y, z])
            
        return points
    '''

    ''' 진행 방향에 대한 탄젠셜 속도 계산 -> cost 계산을 위해'''
    def getting_tangential(self, pos, displacement_dir):
        '''
        wind_dir = self.normalize(self.wind(pos))
        print("Position",pos,"Wind direction", wind_dir)
        V_dir = displacement_dir - wind_dir
        print("V direction is", V_dir)
        if np.hypot(V_dir) != 1:
            print("Normalization is failed.")
        Vel_vector = self.va * V_dir
        return
        '''
        wind = self.wind(pos)                       # 해당 점에서의 바람 성분 (벡터)
        wind_dir = self.normalize(wind)             # 바람의 방향 (정규화)
        wind_intensity = np.linalg.norm(wind)       # 바람의 세기
        #print("Wind Intensity is ",wind_intensity)
        # Even the norm of displacement_dir is 1. 
        alp_i = np.arccos(np.dot(wind_dir, displacement_dir)/(np.linalg.norm(wind_dir) * np.linalg.norm(displacement_dir))) 
        theta_i = self.arcsin_0_pi(wind_intensity * np.sin(alp_i)/ self.va)
        if abs(wind_intensity * np.sin(alp_i)/ self.va) > 1:
            print("wind_intensity * np.sin(alp_i)/ self.va is out of bound [-1,1]")
            return LookupError
        #return self.va * np.dot(V_dir, displacement_dir) + np.linalg.norm(self.wind(pos)) * np.dot(wind_dir, displacement_dir)
        v_tan_i = wind_intensity * np.cos(alp_i) + self.va * np.cos(theta_i)
        
        return v_tan_i

    '''cost 생략 : RRT*에서 쓰이지 않음 (cost 정의 : 도달하는데 걸리는 시간)    
    def cost(self, start, end):
        L0 = self.calc_dist(start,end)                  # 두 점 사이의 직선 거리
        N = 20                                          # 분할 개수
        Cost = 0
        PNTs = self.interpolate_points(start,end,N)     # 두 점 사이를 N개로 분할
        l0 = self.normalize(end.xyz - start.xyz)        # 두 점 사이의 방향 벡터 (정규화)
        
        for i in range(N):
            if self.obs1.collide(PNTs[i]):              # 두 점 사이의 한 점에서 장애물(obs1)과 충돌이 한번이라도 발생하면, cost는 무한
                print("Collision, We will prune it.")
                return np.inf
            Velocity_Tan = self.getting_tangential(PNTs[i,:], l0)
            Cost += L0/N / (Velocity_Tan)
        #print("Cost is ",Cost)
        return Cost
    '''

    '''cost 계산을 위한 휴리스틱 : start 와 end 사이를 분할하여 장애물 충돌을 고려하고 탄젠셜 성분을 통해 cost 계산'''
    def heuristics(self, start, end, n = 5):
        PNTs = self.interpolate_points(start,end,n)         # 경로의 각 점들
        tan_values = []                                     # 각 점들에서의 탄젠셜 성분 리스트
        direction = self.normalize(end.xyz - start.xyz)     # start 에서 goal로 가는 정규화된 방향 벡터
        for i in range(n):                                  # 모든 경로점들에 대해
            if not self.obs1.collide(PNTs[i]):              # 장애물과 충돌 검사 (해당 점이 장애물 내에 속하는지 아닌지)
                tan_value = self.getting_tangential(PNTs[i], direction)      # 충돌이 없으면 탄젠셜 업데이트
                tan_values.append(tan_value)
        if tan_values:                                      # 탄젠셜이 존재하면
            sorted_values = np.sort(tan_values)
            best_case = np.max(tan_values)                  # 가장 큰 탄젠셜 (속도) 반환
            
            return self.calc_dist(start,end)/best_case      # (가장 빨리) 걸리는 시간을 cost 로 반환 (근데 중간에 충돌이 발생해도 발생 안하는 경우만 고려하는데 이래도 되나?)

    '''f_estimated, g_estimated, h_estimated 모두 Prune에서 사용되는 cost'''
    '''f : total cost '''        
    def f_estimated(self, node):
        return self.g_estimated(node) + self.h_estimated(node)

    '''g : cost-to-come '''
    def g_estimated(self, node):
        if self.x_start == node:
            return 0
        return self.heuristics(self.x_start, node)
    
    '''h : cost-to-go '''
    def h_estimated(self, node):
        if self.x_goal == node:
            return 0
        return self.heuristics(node,self.x_goal)
    
    ''' Sample 생략 : RRT*에서 쓰이지 않음 (랜덤 노드 샘플링 (자유 공간 vs Ellipsoid))
    def Sample(self, m, cMax, cMin, xCenter):
        if cMax < np.inf:
            return self.SampleEllipsoid(m, cMax, cMin, xCenter)
        else:
            return self.SampleFreeSpace(m)
    '''

    ''' SampleEllipsoid 생략 : RRT*에서 쓰이지 않음 (노드 랜덤 샘플링 (Ellipsoid))
    def SampleEllipsoid(self, m, cMax, cMin, xCenter):
        if cMax < cMin:
            print("MAX C IS SMALLER THAN MIN C.")
        
        r_old = [cMax / 2.0,
             math.sqrt(cMax ** 2 - cMin ** 2) / 2.0,
             math.sqrt(cMax ** 2 - cMin ** 2) / 2.0]
        
        r = [r_component * self.va for r_component in r_old] 
        #print("r is ", r,"\n")
        L = np.diag(r)
        #print("L is", L)

        ind = 0
        delta = self.delta
        Sample = set()
        while ind < m :
            xBall = self.SampleUnitNBall()
            x_radius_change = L @ xBall.flatten()
            #print("x_radius_change is", x_radius_change)
            x_rand_before_center = self.C @ x_radius_change 
            x_rand_before_center = x_rand_before_center.T
            #print(xCenter)
            x_rand = x_rand_before_center.reshape((3,1)) + xCenter.flatten()
            #print("x_rand is",x_rand,"x_rand before adding center is", x_rand_before_center.reshape((3,1)))
            node = Node([x_rand[(0, 0)], x_rand[(1, 0)], x_rand[(2,0)]])
            in_obs = self.obs1.collide(node.xyz)
            in_x_range = self.x_range[0] + delta <= node.x <= self.x_range[1] - delta
            in_y_range = self.y_range[0] + delta <= node.y <= self.y_range[1] - delta
            in_z_range = self.z_range[0] + delta <= node.z <= self.z_range[1] - delta
            
            if not in_obs and in_x_range and in_y_range and in_z_range:
                Sample.add(node)
                ind += 1
        return Sample
    '''

    ''' SampleFreeSpace 생략 : RRT*에서 쓰이지 않음 (노드 랜덤 샘플링 (FreeSpace)
    def SampleFreeSpace(self, m):
        
        delta = self.delta
        Sample = set()

        ind = 0
        # 자유 공간 : 맵 내 (안전 여유 delta) + 장애물 밖
        while ind < m: 
            node = Node([random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                        random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
                        random.uniform(self.z_range[0] + delta, self.z_range[1] - delta)])
            if self.obs1.collide(node.xyz):
                continue
            else:
                Sample.add(node)
                ind += 1

        return Sample
    '''

    ''' RRT* 용 Sample'''
    def SampleRRT(self):
        
        delta = self.delta

        # 자유 공간 : 맵 내 (안전 여유 delta) + 장애물 밖
        while True: 
            node = Node([random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                        random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
                        random.uniform(self.z_range[0] + delta, self.z_range[1] - delta)])
            if self.obs1.collide(node.xyz):
                continue
            else:
                return node        
    
    ''' Radius 생략 : RRT*에서 쓰이지 않음 (radius 조절)
    def Radius(self, q):
        
        cBest = self.g_T[self.x_goal]
        
        lambda_X = len([1 for v in self.Tree.V if self.f_estimated(v) <= cBest])
        #print("Lambda_X and q are ",lambda_X,q)
        radius = 2 * self.eta * (1.5 * lambda_X / math.pi * math.log(q) / q) ** 0.5
        print("radius is ",radius) 
        
        return radius
    '''

    ''' ExpandVertex 생략 : RRT*에서 쓰이지 않음
    def ExpandVertex(self, v):
        # QV에 있는 거 지우면서 EXPAND 시작하기.
        self.Tree.QV.remove(v) 
        # X_near은 Sample들 중에서 radius r 안에 들어오는 모든 X_samples. 
        #print("X_sample's Number",len(self.X_sample))
        X_near = {x for x in self.X_sample if self.calc_dist(x, v)/self.va <= self.Tree.r}
        #print("X_near's Number",len(X_near))
        
        for x in X_near:
            # estimated 는 어떻게 얻는걸까?
            # self.start로부터의 직선거리. <- g // self.goal까지의 거리 : h_estimated . 
            # TEST 1
            #print("the current node value is ",self.g_estimated(v) + self.calc_dist(v, x)/self.va + self.h_estimated(x),"the bar is ", self.g_T[self.x_goal])
            if self.g_estimated(v) + self.calc_dist(v, x)/self.va + self.h_estimated(x) < self.g_T[self.x_goal]:
                self.g_T[x] = np.inf
                self.Tree.QE.add((v, x))

        if v not in self.Tree.V_old:
            # self.Tree.r 은 Time cost!!!! - July 6th.
            V_near = {w for w in self.Tree.V if self.calc_dist(w, v)/self.va <= self.Tree.r}
            for w in V_near:
                # TEST 2
                if (v, w) not in self.Tree.E and \
                        self.g_estimated(v) + self.calc_dist(v, w)/self.va + self.h_estimated(w) < self.g_T[self.x_goal] and \
                        self.g_T[v] + self.calc_dist(v, w)/self.va < self.g_T[w]:
                    self.Tree.QE.add((v, w))
                    if w not in self.g_T:
                        self.g_T[w] = np.inf
    '''
    
    '''  BestVertexQueueValue 생략 : RRT*에서 쓰이지 않음
    def BestVertexQueueValue(self):
        if not self.Tree.QV:
            return np.inf

        return min(self.g_T[v] + self.h_estimated(v) for v in self.Tree.QV)
    '''
    
    '''  BestEdgeQueueValue 생략 : RRT*에서 쓰이지 않음
    def BestEdgeQueueValue(self):
        if not self.Tree.QE:
            return np.inf

        return min(self.g_T[v] + self.calc_dist(v, x)/self.va + self.h_estimated(x)
                   for v, x in self.Tree.QE)
    '''
    
    '''  BestInVertexQueue 생략 : RRT*에서 쓰이지 않음
    def BestInVertexQueue(self):
        if not self.Tree.QV:
            print("QV is Empty!")
            return None
        # {key : value}
        v_value = {v: self.g_T[v] + self.h_estimated(v) for v in self.Tree.QV}

        return min(v_value, key=v_value.get)
    '''
    
    '''  BestInEdgeQueue 생략 : RRT*에서 쓰이지 않음
    def BestInEdgeQueue(self):
        if not self.Tree.QE:
            print("QE is Empty!")
            return None

        e_value = {(v, x): self.g_T[v] + self.calc_dist(v, x)/self.va + self.h_estimated(x)
                   for v, x in self.Tree.QE}

        return min(e_value, key=e_value.get)
    '''
    
    '''  SampleUnitNBall 생략 : RRT*에서 쓰이지 않음
    @staticmethod
    def SampleUnitNBall():
        while True:
            x, y, z = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1,1)
            if x ** 2 + y ** 2 + z ** 2 < 1:
                return np.array([[x], [y], [z]]).T
    '''
    
    ''' arcsin을 0 ~ pi 값으로 반환 (np.arcsin은 -pi/2 ~ pi/2 값을 반환) '''
    @staticmethod
    def arcsin_0_pi(x):
        arcsin_value = np.arcsin(x)
        if arcsin_value < 0:
            return arcsin_value + np.pi
        return arcsin_value        
    
    ''' 두 점 사이를 일정 개수만큼 분할 '''
    @staticmethod
    # estimation part should be changed. Not only distance but also energy cost.
    def interpolate_points(point1, point2, num_points):

        x_values = np.linspace(point1.x, point2.x, num_points)
        y_values = np.linspace(point1.y, point2.y, num_points)
        z_values = np.linspace(point1.z, point2.z, num_points)
        return np.vstack((x_values, y_values,z_values)).T   

    ''' 벡터 정규화 '''        
    @staticmethod
    def normalize(v):
        return v/np.linalg.norm(v) if np.linalg.norm(v) !=0 else v
    
    ''' RotationToWorldFrame 생략 : RRT*에서 쓰이지 않음 (SampleEllipsoid 에서 사용되는 것임)
    @staticmethod
    def RotationToWorldFrame(x_start, x_goal, L):
        # L means cMin.
        # 
        dst = np.array([[(x_goal.x - x_start.x) / L],
                        [(x_goal.y - x_start.y) / L], 
                        [(x_goal.z - x_start.z) / L]])
        src = np.array([[1.0], [0.0], [0.0]])
        # @ : matrix multiplication using the "@" operator in Numpy.
        M = np.outer(dst,src)
        # Eigen value, Eigen vector, 
        # To find the optimal rotation between two 3D vectors.
        U, _, V_T = np.linalg.svd(M, True, True)
        #C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T
        C = np.dot(U,V_T)
        
        return C
    '''

    ''' 두 점 사이의 직선 거리 반환 (3차원) '''
    @staticmethod
    def calc_dist(start, end):
        return math.hypot(start.x - end.x, start.y - end.y, start.z - end.z)

    ''' 두 점의 x,y 사이의 거리 및 각도 반환 (2차원) '''
    @staticmethod
    def calc_dist_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        dz = node_end.z - node_start.z
        return math.hypot(dx, dy), math.atan2(dy, dx)

    ''' animation 생략 : RRT*에서 쓰이지 않음 (BIT용 애니메이션) 
    def animation(self, xCenter, cMax, cMin):
        
        if self.text is None:
                self.text = self.ax.text2D(0.05, 0.95, "", fontsize=15, transform=self.ax.transAxes, verticalalignment='top')

        try:
            
            self.fig.canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            for v in self.X_sample:
                if not self.flagF:
                    #print("Ellipsoid")
                    self.ax.scatter(v.x, v.y, v.z, marker='.', color='grey', s = 1)
                if self.flagF:
                    self.ax.scatter(v.x, v.y, v.z, marker='.', color='black', s = 1)
            if cMax < np.inf:
                # TODO : Changing 3D
                self.draw_ellipse(self.ax, xCenter, cMax, cMin)

            for v, w in self.Tree.E:
                self.ax.plot([v.x, w.x], [v.y, w.y], [v.z, w.z],'-g',linewidth = 2, alpha = 0.5)
            path_cost = self.g_T[self.x_goal]
            path_text = f"Cost: {path_cost}"
            self.text.set_text(path_text)
            #self.ax.text2D(0.05, 0.95, path_text, fontsize=15, transform=self.ax.transAxes, verticalalignment='top')
            plt.title("Wind Aware Batch Informed Trees")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            self.fig.savefig("error_figure.png")
            raise
    '''

    ''' draw_ellipse 생략 : RRT*에서 쓰이지 않음
    def draw_ellipse(self, ax, x_center, c_best, dist):
        #TODO : Eplliipsoid 표현해줘.
        
        # dist means cMin
        # c_best means cMax : by the tree.
        a = c_best / 2.0 * self.va
        b = math.sqrt(c_best ** 2 - dist ** 2) / 2.0 * self.va
        c = math.sqrt(c_best ** 2 - dist ** 2) / 2.0 * self.va
        
        #angle = math.pi / 2.0 - theta
        cx = x_center[0] 
        cy = x_center[1]
        cz = x_center[2]
        
        t = np.arange(0, 2 * math.pi + 0.1, 0.2)
        phi = np.arange(0, 2 * math.pi + 0.1, 0.2)

        x = [a * math.sin(iphi) * math.cos(it)  for it in t for iphi in phi]
        y = [b * math.sin(iphi) * math.sin(it)  for it in t for iphi in phi]
        z = [c * math.cos(iphi) for iphi in phi]
        
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        
        z = np.tile(z,len(t))
        #rot = Rot.from_euler('x', -angle).as_matrix()
        rot = self.C
        fx = rot @ np.array([x, y, z])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        pz = np.array(fx[2, :] + cz).flatten()
        #print("The center is", cx, cy, cz)
        print("The radius of each coordinate is ",a,b,c)
        self.ax.scatter(cx, cy, cz, marker='.', color='blue', s = 6)
        self.ax.plot(px, py, pz, linestyle='--', color='darkorange', linewidth=0.25)
    '''   

    def plot_tree(self):
        # 트리의 각 노드의 부모를 따라 그리기
        for v in self.Tree.V:
            if v.parent is not None:  # 부모가 있을 때만
                parent = v.parent
                self.ax.scatter(v.x, v.y, v.z, marker = 'o' ,color = 'k',s = 5)
                self.ax.plot([v.x, parent.x], [v.y, parent.y], [v.z, parent.z], 'g-')     

        # # 시작점과 목표점을 강조해서 그리기
        # self.ax.scatter(self.x_start.x, self.x_start.y, self.x_start.z, color='r', s=100, label="Start")  # 빨간색 점
        # self.ax.scatter(self.x_goal.x, self.x_goal.y, self.x_goal.z, color='g', s=100, label="Goal")    # 초록색 점

    plt.legend()

'''MAIN 실행 함수'''
def main():

    # TODO : 
    x_start = (0.0, 0.0, 10.0)  # Starting node
    x_goal = (3000, 3000,3000)  # Goal node
    print("Start point is ", x_start)
    print("Goal point is ", x_goal)

    r = 100 # search radius
    iter_max = 200 
    va = 20 
    ResolutionType = 'normal'

    step_len = 50
    
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


    rrt = RRTStar(x_start, x_goal, r, step_len, iter_max, va, u, v, w)
    #bit.draw_things()
    rrt.planning()

    #bit = BITStar(x_start, x_goal, eta, iter_max)
    #bit.animation("Batch Informed Trees (BIT*)")
    
if __name__== '__main__':
    main()
