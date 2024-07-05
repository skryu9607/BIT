import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from noise import pnoise2
from pytictoc import TicToc

"""
Code 정보
1. 수평풍 : Perlin Noise 를 이용 (wind_model_v1 과의 차이점)
2. 수직풍 : leaning 을 고려한 Allen 모델 (논문 수식이 아닌 matlab 기준) (wind_model_v2 와의 차이점)
3. 수직풍을 고려할 때, 수평풍 Wx, Wy를 수정 (사전 정의해야 하는 파라미터 -> 상승기류 내부 영역에 존재하는 수평풍 u,v 의 평균) (wind_model_v2 와의 차이점)
4. Resolution : x-y-z 모두 0.1로 통일 & Plot 제거 (wind_model_v3 와의 차이점)
"""

"""
Plot 정보
1. 특정 고도에서의 2차원 horiziontal wind 분포 (수평 x-y 방향만 고려)
2. 특정 고도에서의 thermal updraft 속도 분포 (수직 z 방향만 고려)
3. 수평 및 수직 방향 바람장 plots
"""

######################################################################################################

# Perlin 노이즈를 사용하여 2D wind field map 생성

def generate_wind_field(num_x, num_y, num_z, scale, wind_speed, seed):
    np.random.seed(seed)

    # u = np.zeros((num_x, num_y), dtype=np.float64)
    # v = np.zeros((num_x, num_y), dtype=np.float64)
    u = np.zeros((num_x, num_y, num_z), dtype=np.float64)
    v = np.zeros((num_x, num_y, num_z), dtype=np.float64)
    
    for i in range(num_x):
        for j in range(num_y):
            
            noise1 = pnoise2(i * scale, j * scale, octaves=1, base=seed)
            noise2 = pnoise2(i * scale, j * scale, octaves=1, base=seed + 10)

            angle     = noise1 * 2 * np.pi
            magnitude = noise2 * wind_speed + wind_speed

            # # 고도 (k) 에 따른 수평풍 변화는 없음
            for k in range(num_z):    
                u[i, j, k] = magnitude * np.cos(angle)
                v[i, j, k] = magnitude * np.sin(angle)
    
    return u, v

######################################################################################################

# Thermal Model Function

def allen_model(z, zi, w_star, map_x, map_y, num_x, num_y, u, v, xc=0, yc=0, u0=0, v0=0):
    
    # 상승기류 외곽 반지름 계산 [m]
    r2 = max(10, 0.102 * (z / zi)**(1/3) * (1 - 0.25 * z / zi) * zi )
    
    # 평균 상승기류 속도 계산 (Lenschow 방정식) [m/s]
    w_avg = 1.0 * (z / zi)**(1/3) * (1 - 1.1 * z / zi) * w_star    
    
    # r1/r2 ratio 결정
    if r2 < 600:
        r1r2 = 0.0011*r2 +0.14
    else:
        r1r2 = 0.8

    # 상승기류 코어 반지름 계산
    r1 = r1r2 * r2

    # 상승기류 최대 속도 계산 (wc)
    w_peak = 3 * w_avg * ((r2**2)*(r2-r1)) / (r2**3 - r1**3)

    # 그리드 간격
    grid_x = 2 * map_x / num_x
    grid_y = 2 * map_y / num_y

    # (d/2) 가 각 축에 대해 몇 개의 grid로 이루어지는지
    num_grid_x = int((r2) // grid_x)
    num_grid_y = int((r2) // grid_y)
    
    # thermal 중심 (xc,yc)이 맵 중심 (0,0)에 위치할 때, 각 축의 인덱스 상/하한 (thermal 내부에 존재하는 수평풍에 대해) 
    idx_x_low = int(num_x / 2 - 1 - num_grid_x)
    idx_x_upp = int(num_x / 2 - 1 + num_grid_x)
    idx_y_low = int(num_y / 2 - 1 - num_grid_y)
    idx_y_upp = int(num_y / 2 - 1 + num_grid_y)

    # Wx, Wy 는 해당 고도의 상승기류 내부 영역에 존재하는 수평풍 (u,v) 의 평균!!
    Wx = 0
    Wy = 0
    
    for idx_x in range(idx_x_low, idx_x_upp):
        for idx_y in range(idx_y_low, idx_y_upp):
            # Wx = Wx + u[idx_x,idx_y]
            # Wy = Wy + v[idx_x,idx_y]
            Wx = Wx + u[idx_x,idx_y,0]
            Wy = Wy + v[idx_x,idx_y,0]
    
    Wx = Wx / ((2*num_grid_x + 1)*(2*num_grid_y + 1))
    Wy = Wy / ((2*num_grid_x + 1)*(2*num_grid_y + 1))

    # Bencatel 모델 적용: 상승기류 중심선 위치 계산 [m]
    xt = xc + (Wx - u0) * z / w_avg
    yt = yc + (Wy - v0) * z / w_avg

    # 2D 좌표 배열 생성
    x = np.linspace(-map_x, map_x, num_x).astype(np.float64)
    y = np.linspace(-map_y, map_y, num_y).astype(np.float64)
    X, Y = np.meshgrid(x, y)

    # 상승기류 중심에서의 거리 계산 [m0] (tuple)
    r = np.sqrt((X - xt)**2 + (Y - yt)**2)
    
    # 벨 모양 곡선 형상 상수
    if z < zi:
        if r1r2 < (0.14+0.25)/2:
            K = [1.5352, 2.5826, -0.0113, 0.0008]
        elif r1r2 < (0.25+0.36)/2:
            K = [1.5265, 3.6054, -0.0176, 0.0005]
        elif r1r2 < (0.36+0.47)/2:
            K = [1.4866, 4.8354, -0.0320, 0.0001]
        elif r1r2 < (0.47+0.58)/2:
            K = [1.2042, 7.7904,  0.0848, 0.0001]
        elif r1r2 < (0.58+0.69)/2:
            K = [0.8816, 13.972,  0.3404, 0.0001]
        elif r1r2 < (0.69+0.80)/2:
            K = [0.7067, 23.994,  0.5689, 0.0002]
        else:
            K = [0.6189, 42.797,  0.7157, 0.0001]

        k1 = K[0]
        k2 = K[1]
        k3 = K[2]
        k4 = K[3]
        
        ws = (1 / (1 + abs(k1 * r / r2 + k3)**k2)) + k4 * (r / r2)
    else:
        ws = 0

    w_l = np.empty((num_x, num_y), dtype=np.float64)
    w_d = np.empty((num_x, num_y), dtype=np.float64)
    swd = np.empty((num_x, num_y), dtype=np.float64)
    for ii in range(num_x):
        for jj in range(num_y):
            if r[ii,jj]>r1 and (r[ii,jj]/r2)<2:
                w_l[ii,jj] = np.pi/6*np.sin(np.pi*r[ii,jj]/r2)     #downdraft, positive up (논문과 달리 부호가 +임)
            else:
                w_l[ii,jj] = 0
    
            if (z/zi)>0.5 and (z/zi)<0.9:
                swd[ii,jj] = 2.5*((z/zi)-0.5)
                w_d[ii,jj] = swd[ii,jj]*w_l[ii,jj]
            else:
                swd[ii,jj] = 0
                w_d[ii,jj] = 0
    # print(w_d)        
    
    w = w_peak * (ws + w_d)         # 논문 수식과 다름
    
    # 해당 영역(S) 에서의 thermal 최대 개수 
    S = (2*map_x)*(2*map_y)
    N = math.floor(0.6*S/(zi*r2))
    
    # 환경 하강 속도 (environment sink velocity) (논문 수식과 다름)
    w_e = np.empty((num_x, num_y), dtype=np.float64)
    w_total = np.empty((num_x, num_y), dtype=np.float64)
    for ii in range(num_x):
        for jj in range(num_y):
            w_e[ii,jj] = -(w_avg*N*np.pi*r2**2*(1-swd[ii,jj]))/(S-N*np.pi*r2**2)
            # print(w_e)
            
            # 총 상승기류 속도
            if r[ii,jj]>r1:
                w_total[ii,jj] = w[ii,jj] * (1 - w_e[ii,jj] / w_peak) + w_e[ii,jj]
            else:
                w_total[ii,jj] = w[ii,jj]

    return w

######################################################################################################

t = TicToc()
t.tic()
# Thermal 파라미터 설정
zi = 4000           # 대류 혼합층 두께 [m]
w_star = 10         # 상승기류 속도 스케일 [m/s]
xc = 0              # 상승기류 중심의 x좌표
yc = 0              # 상승기류 중심의 y좌표

# 맵 설정 [m] : x, y 축 절반 길이 (맵 중심은 (0,0))
map_x = 3000
map_y = 3000

# Resolution 설정 (개수)
resolution = 10
num_x = int((2 * map_x / resolution) + 1)
num_y = int((2 * map_x / resolution) + 1)
num_z = int(((0.9 * zi - resolution) / resolution) + 1)
altitudes = np.linspace(resolution, 0.9*zi, num_z).astype(np.float64)

# Horizontal Wind 파라미터 설정
w_hor = 1         # 수평풍 속도 스케일  [m/s]
scale = 0.1         # Perlin noise scale
seed = 1            # seed number

# 3차원 quiver의 길이 파라미터 (map 크기에 따라 보기 좋게 수정해주는 역할)
# arrow_size = 30

######################################################################################################
# Plot 1
######################################################################################################

X, Y, Z = np.meshgrid(np.linspace(-map_x, map_x, num_x).astype(np.float64), np.linspace(-map_y, map_y, num_y).astype(np.float64), altitudes)

# Horizontal Wind Vector Field 
u, v = generate_wind_field(num_x, num_y, num_z, scale, w_hor, seed)
print(type(u[0,0]))
print(u.itemsize)

# # Horizontal Wind Plot
# ax = plt.axes()

# wind = ax.quiver(X[:,:,0], Y[:,:,0], u[:,:,0], v[:,:,0], np.sqrt(u[:,:,0]**2 + v[:,:,0]**2), cmap=cm.winter.reversed())
# plt.colorbar(wind, ax=ax, orientation='vertical').set_label('Wind Speed (m/s)')

# # Set title and labels
# plt.title('Wind Vector Map (2D)')
# plt.xlabel('X [m]')
# plt.ylabel('Y [m]')

# # Show the plot
# plt.show()

######################################################################################################
# Plot 2
######################################################################################################

# Vertical Wind (Thermal Updraft)
w = np.empty((num_x, num_y, num_z), dtype=np.float64)

for i, z in enumerate(altitudes):
    
    w_vert = allen_model(z, zi, w_star, map_x, map_y, num_x, num_y, u, v)
    w[:,:,i] = w_vert

t.toc()

np.save('C:/Users/seung/WindData/u',u)
np.save('C:/Users/seung/WindData/v',v)
np.save('C:/Users/seung/WindData/w',w)
        
# # 3D Surface Plot in a specific altitude 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X[:,:,round(0.6*num_z)], Y[:,:,round(0.6*num_z)], w[:,:,round(0*num_z)], cmap=cm.winter.reversed())
# fig.colorbar(surf, label="Updraft Velocity (m/s)")
# ax.set_xlabel("X (m)")
# ax.set_ylabel("Y (m)")
# ax.set_zlabel("Updraft Velocity (m/s)")
# plt.title("Updraft Thermal Model with Leaning")
# plt.show()

# ######################################################################################################
# # Plot 3
# ######################################################################################################

# # Wind speed [m/s]
# s = np.sqrt(u**2 + v**2 + w**2)

# # Figure and Plot Settings

# norm = Normalize(vmin=np.min(s), vmax=np.max(s))
# cmap = cm.winter.reversed()

# X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
# u_flat, v_flat, w_flat = u.flatten(), v.flatten(), w.flatten()
# s_flat = s.flatten()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Calculate colors for each arrow
# colors = cmap(norm(s_flat))

# # Plot each quiver individually with corresponding color
# for i in range(len(X_flat)):
#     # 1. u,v,w 방향 모두
#     ax.quiver(X_flat[i], Y_flat[i], Z_flat[i], u_flat[i], v_flat[i], w_flat[i], color=colors[i], length=arrow_size)
#     # 2. w 방향만
#     # ax.quiver(X_flat[i], Y_flat[i], Z_flat[i], 0, 0, w_flat[i], color=colors[i], length=arrow_size)
    

# # Create color bar
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array(s_flat)
# fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)

# # Set title and labels
# plt.title('Wind Vector Map')
# plt.xlabel('X [m]')
# plt.ylabel('Y [m]')

# # Show the plot
# plt.show()
