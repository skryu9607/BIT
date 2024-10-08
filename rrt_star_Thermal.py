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
        self.step_len = step_len           # 한 스텝당 갈 수 있는 최대 거리
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
        self.fig = plt.figure(figsize=(12,9))
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
        abc = np.array([600.,600.,600.])
        shape = np.array([1.,1.,2.])
        self.obs1 = Obstacles(xyz0,abc,shape,self.x_range,self.y_range,self.z_range)
        #self.obs1.draw(self.ax,self.x_start,self.x_goal)
        
    '''PATH PLANNING : RRT*'''
    def planning(self):

        # Initialize for visualization
        self.ax.view_init(elev=20, azim=-85)
        self.ax.scatter(self.x_start.x,self.x_start.y,self.x_start.z,marker = 's' ,color = 'blue',s = 50)
        self.ax.scatter(self.x_goal.x,self.x_goal.y,self.x_goal.z,marker = 'x' ,color = 'blue',s = 50)
        
        # Initialize for data storing
        file_path = "data_rrt.json"
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
        cnt = 0         # 노드의 총 개수

        for k in range(self.iter_max):
            
            # 노드 랜덤 샘플링 (x_rand)
            x_rand = self.SampleRRT()
            
            # Nearest 노드 찾기 (x_nearest)
            x_nearest = self.nearest(x_rand)
            
            # New 노드 생성 (x_new) (step_len 고려해서) (충돌이 없으면, parent = x_nearest, 충돌이 있으면 parent = None)
            x_new = self.make_new(x_nearest, x_rand)
            
            if x_new.parent is None:                        # x_new.parent 가 None 이면, 다음 샘플링 루프로 넘김 (다시 샘플링)
                continue
            
            cnt +=1
            print(f"Iteration {k} & No(nodes) {cnt}")
            self.Tree.V.add(x_new)                          # Tree에 x_new 추가

            # self.g_T[x_new] = self.g_estimated(x_nearest) + self.calc_dist(x_nearest, x_new)/self.va    # x_nearest를 통해 x_new 까지의 cost-to-come(g)
            self.g_T[x_new] = self.g_T[x_nearest] + self.cost(x_nearest, x_new)
            
            # 1. x_new의 parent 재선정
            near_nodes = self.find_near_nodes(x_new)
            for x_near in near_nodes:
                if self.collisionFree(x_near, x_new):   # x_near 부터 x_new 로의 경로 상에 충돌이 없는 경우 (feasible path 인 경우)
                    # g_near = self.g_estimated(x_near) + self.calc_dist(x_near,x_new)/self.va
                    g_near = self.g_T[x_near] + self.cost(x_near,x_new)
                    
                    if g_near < self.g_T[x_new]:
                        x_new.parent = x_near
                        self.g_T[x_new] = g_near

            # 2. Rewiring (기존 노드들의 parent 재선정)
            self.rewire(x_new, near_nodes)
            
            # x_new 노드가 goal 노드 일정 반경 내에 들어오고, x_goal까지 feasible 하면
            if self.calc_dist(x_new, self.x_goal) <= self.step_len and self.collisionFree(x_new, self.x_goal):
                self.x_goal.parent = x_new
                self.g_T[self.x_goal] = self.g_T[x_new] + self.cost(x_new,self.x_goal)
                self.Tree.V.add(self.x_goal)

            # Goal 에 도달한 경우, Path 추출 및 반복문 종료
            if self.x_goal.parent is not None:
                
                path_x, path_y, path_z = self.ExtractPath()     # Path 추출
                self.Flag = True                                # Success 여부
                break
        
        print('x_goal.parent ',self.x_goal.parent)

        if not self.Flag:
            print(f"Solution Not Found within {self.iter_max} iteration.")
        else:
            print(f"Solution Found in {k}th iteration!")
            print(f"Path's cost : {self.g_T[self.x_goal]}")

        # Tree Plot
        self.plot_tree()

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

        # Path Plot
        if self.Flag:
            # Plot
            # self.ax.title("Wind Aware Batch Informed Trees")
            # self.ax.xlabel("X")
            # self.ax.ylabel("Y")
            # plt.zlabel("Z")
            self.ax.plot(path_x, path_y, path_z , linewidth=2, color='r',linestyle ='--')
        plt.show()    

    ''' Tree 에서 node 와 가장 가까운 노드 (x_nearest) 찾기 '''
    def nearest(self, node):
        min_dist = np.inf
        for v in self.Tree.V:
            dist = self.calc_dist(node, v)
            if dist < min_dist:
                min_dist = dist
                x_nearest = v
        
        return x_nearest
    
    ''' 새로운 노드 (x_new) 생성 '''
    def make_new(self, x_nearest, x_rand):
        distance = self.calc_dist(x_nearest, x_rand)
        if distance > self.step_len:
            direction = self.normalize(x_rand.xyz - x_nearest.xyz)
            x_new = Node((x_nearest.x + direction[0] * self.step_len,
                             x_nearest.y + direction[1] * self.step_len,
                             x_nearest.z + direction[2] * self.step_len))
            if self.collisionFree(x_nearest, x_new):          # 경로 상에 장애물과 충돌이 없으면, 부모 노드 추가
                x_new.parent = x_nearest
            else:
                x_new.parent = None
            return x_new
        else:
            if self.collisionFree(x_nearest, x_rand):
                x_rand.parent = x_nearest
            else:
                x_rand.parent = None     
            return x_rand

    ''' node 와 특정 반경 r 내에 위치한 가까운 노드들 반환 '''
    def find_near_nodes(self, node):
        near_nodes = set()
        for v in self.Tree.V:
            dist = self.calc_dist(node, v)
            if dist < self.Tree.r:
                near_nodes.add(v)
        near_nodes.remove(node)     # 본인은 이웃노드에서 제외해야 함
        return near_nodes

    def rewire(self, x_new, near_nodes):
        for node in near_nodes:
            # if self.collisionFree(x_new, node) and self.g_estimated(x_new) + self.calc_dist(x_new, node)/self.va < self.g_estimated(node):
            if self.collisionFree(x_new, node) and self.g_T[x_new] + self.cost(x_new, node) < self.g_T[node]:
                node.parent = x_new
                # self.g_T[node] = self.g_estimated(x_new) + self.calc_dist(x_new, node)/self.va
                self.g_T[node] = self.g_T[x_new] + self.cost(x_new, node)
    
    def collisionFree(self, from_node, to_node, n = 10):
        PNTs = self.interpolate_points(from_node,to_node,n)     # 경로의 각 점들
        for i in range(n):                                      # 모든 경로점들에 대해
            if self.obs1.collide(PNTs[i]):                      # 장애물과 충돌 검사 (해당 점이 장애물 내에 속하는지 아닌지)
                return False                                    # 하나라도 충돌하면 False (Collison free 하지 않다)
        return True                                             # 충돌이 없으면 True
    
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

    '''해당 위치에서의 바람 성분 반환'''    
    def wind(self, points):
        #print(points)
        wind_at_node = wind_catcher(points[0],points[1],points[2],self.u,self.v,self.w)
        #print(wind)
        return wind_at_node
        #return [0.,0.,0.]


    ''' 진행 방향에 대한 탄젠셜 속도 계산 -> cost 계산을 위해'''
    def getting_tangential(self, pos, displacement_dir):

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

    '''cost 정의 : 도달하는데 걸리는 시간 '''   
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
   
    @staticmethod
    def normalize(v):
        return v/np.linalg.norm(v) if np.linalg.norm(v) !=0 else v
    

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
    ''' Tree Plot 함수 '''
    def plot_tree(self):
        # 트리의 각 노드의 부모를 따라 그리기
        for v in self.Tree.V:
            if v.parent is not None:  # 부모가 있을 때만
                px, py, pz = v.parent.x, v.parent.y, v.parent.z
                self.ax.scatter(v.x, v.y, v.z, marker = 'o' ,color = 'k', s=1)
                self.ax.plot([v.x, px], [v.y, py], [v.z, pz], 'g-')     

    ''' Tree 완전성 체크 함수 : 모든 노드에 대해 부모 노드를 역추적하여 시작 노드에 도달하는지 확인'''
    def check_tree_integrity(self):
        
        for node in self.Tree.V:

            current_node = node
            path_trace = []
            
            while current_node.parent is not None:
                path_trace.append(current_node)
                current_node = current_node.parent
            
            # 시작 노드에 도달하지 못했다면 경고 출력
            if current_node != self.x_start:
                print(f"Node {node} does not connect to the start node. Issue found!")
                print(f"Trace: {[str(n) for n in path_trace]}")
                return False
        
        print("All nodes correctly connect to the start node.")
        return True


'''MAIN 실행 함수'''
def main():

    # TODO : 
    x_start = (0.0, 0.0, 10.0)      # Starting node
    x_goal = (3000, 3000, 3000)     # Goal node
    # x_goal = (1000, 1000, 1000)     # Goal node
    print("Start point is ", x_start)
    print("Goal point is ", x_goal)

    r = 500 # search radius
    iter_max = 5000
    va = 20 
    ResolutionType = 'normal'

    step_len = 300
    
    # Wind Data Path : 승걸
    onedrive_path = '/Users/seung/WindData/'
    # Wind Data Path : 민조
    # onedrive_path = 'C:/Users/LiCS/Documents/MJ/KAIST/Paper/2025 ACC/Wind Energy/Code/windData/'
    
    #Mac : OneDrive
    u = np.load(f'{onedrive_path}/{ResolutionType}/u_{ResolutionType}.npy')
    v = np.load(f'{onedrive_path}/{ResolutionType}/v_{ResolutionType}.npy')
    w = np.load(f'{onedrive_path}/{ResolutionType}/w_{ResolutionType}.npy')
    
    '''
    #Windows
    u = np.load(f'{onedrive_path}u_{ResolutionType}.npy')
    v = np.load(f'{onedrive_path}v_{ResolutionType}.npy')
    w = np.load(f'{onedrive_path}w_{ResolutionType}.npy')
    '''

    rrt = RRTStar(x_start, x_goal, r, step_len, iter_max, va, u, v, w)
    #bit.draw_things()
    rrt.planning()
    rrt.check_tree_integrity()

    #bit = BITStar(x_start, x_goal, eta, iter_max)
    #bit.animation("Batch Informed Trees (BIT*)")
    
if __name__== '__main__':
    main()
