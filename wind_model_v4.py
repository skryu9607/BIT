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
4. Resolution : x-y-z 모두 같은 값로 통일 & Plot 제거 (wind_model_v3 와의 차이점)
"""

"""
Plot 정보
1. 특정 고도에서의 2차원 horiziontal wind 분포 (수평 x-y 방향만 고려)
2. 특정 고도에서의 thermal updraft 속도 분포 (수직 z 방향만 고려)
3. 수평 및 수직 방향 바람장 plots
"""

######################################################################################################

# Perlin 노이즈를 사용하여 2D wind field map 생성

def generate_wind_field(num_x, num_y, scale, wind_speed, seed):
    np.random.seed(seed)

    u = np.zeros((num_x, num_y))
    v = np.zeros((num_x, num_y))
    
    for i in range(num_x):
        for j in range(num_y):
            
            noise1 = pnoise2(i * scale, j * scale, octaves=1, base=seed)
            noise2 = pnoise2(i * scale, j * scale, octaves=1, base=seed + 10)

            angle     = noise1 * 2 * np.pi
            magnitude = noise2 * wind_speed + wind_speed

            # 고도 (k) 에 따른 수평풍 변화는 없음
            for k in range(num_z):    
                u[i, j] = magnitude * np.cos(angle)
                v[i, j] = magnitude * np.sin(angle)
    
    return u, v

######################################################################################################

# Thermal Model Function

def allen_model(resolution,z, zi, w_star, x_min, x_max, y_min, y_max, u, v, xc, yc, xt_old, yt_old, u0=0, v0=0):
    
    # 맵 데이터 파라미터
    x_len = x_max - x_min
    y_len = y_max - y_min

    # Resolution 설정 (개수)
    num_x = int((x_len / resolution) + 1)
    num_y = int((y_len / resolution) + 1)
    
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

    # (d/2) 가 각 축에 대해 몇 개의 grid로 이루어지는지
    num_grid = int((r2) // resolution)
    
    # xt_old, yt_old 의 인덱스
    xt_old_idx = int((xt_old - x_min)/resolution)
    yt_old_idx = int((yt_old - y_min)/resolution)

    # 2 * r2 의 정사각형 (중심은 xt_old, yt_old)에 해당하는 영역에 대한 인덱스 상/하한 -> u, v에 대한 인덱싱을 위해
    idx_x_low = int(xt_old_idx - num_grid)
    idx_x_upp = int(xt_old_idx - num_grid)
    idx_y_low = int(yt_old_idx - num_grid)
    idx_y_upp = int(yt_old_idx - num_grid)

    # Wx, Wy 는 해당 고도의 상승기류 내부 영역에 존재하는 수평풍 (u,v) 의 평균!!
    Wx = 0  # 초기화
    Wy = 0  # 초기화
    for idx_x in range(idx_x_low, idx_x_upp):
        for idx_y in range(idx_y_low, idx_y_upp):
            Wx = Wx + u[idx_x,idx_y]
            Wy = Wy + v[idx_x,idx_y]
    
    Wx = Wx / ((2*num_grid + 1)*(2*num_grid + 1))
    Wy = Wy / ((2*num_grid + 1)*(2*num_grid + 1))

    # Bencatel 모델 적용: 상승기류 중심선 위치 (xt,yt) 계산 [m]   * (xc,yc) : 지표면에서의 중심 좌표, (u0, v0) : thermal의 수평 이동 속도
    xt = xc + (Wx - u0) * z / w_avg
    yt = yc + (Wy - v0) * z / w_avg

    # 2D 좌표 배열 생성
    x = np.linspace(x_min, x_max, num_x)
    y = np.linspace(y_min, y_max, num_y)
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

    w_l = np.empty((num_x, num_y))
    w_d = np.empty((num_x, num_y))
    swd = np.empty((num_x, num_y))
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
    S = (x_len)*(y_len)
    N = math.floor(0.6*S/(zi*r2))
    
    # 환경 하강 속도 (environment sink velocity) (논문 수식과 다름)
    w_e = np.empty((num_x, num_y))
    w_total = np.empty((num_x, num_y))
    for ii in range(num_x):
        for jj in range(num_y):
            w_e[ii,jj] = -(w_avg*N*np.pi*r2**2*(1-swd[ii,jj]))/(S-N*np.pi*r2**2)
            # print(w_e)
            
            # 총 상승기류 속도
            if r[ii,jj]>r1:
                w_total[ii,jj] = w[ii,jj] * (1 - w_e[ii,jj] / w_peak) + w_e[ii,jj]
            else:
                w_total[ii,jj] = w[ii,jj]

    return w, xt, yt

######################################################################################################

t = TicToc()
t.tic()

# Thermal 파라미터 설정
zi = 4000           # 대류 혼합층 두께 [m]
w_star = 10         # 상승기류 속도 스케일 [m/s]
xc = 1800            # 상승기류 중심의 x좌표 (지표면) (맵 중앙)
yc = 2400            # 상승기류 중심의 y좌표 (지표면) (맵 중앙)

# 맵 설정 [m] : x, y 축 절반 길이 (맵 중심은 (0,0))
x_min = -1000
x_max = 5000
y_min = -1000
y_max = 5000

x_len = x_max - x_min
y_len = y_max - y_min

# Resolution 설정 (개수)
ResolutionType = 'high'
# Coarse 
if ResolutionType == 'coarse':
    resolution = 20
elif ResolutionType == 'normal':
    resolution = 10
elif ResolutionType == 'high':
    resolution = 5
elif ResolutionType == 'highest':
    resolution = 1
num_x = int((x_len / resolution) + 1)
num_y = int((y_len / resolution) + 1)
num_z = int(((0.9 * zi - resolution) / resolution) + 1)
altitudes = np.linspace(resolution, 0.9*zi, num_z)

# Horizontal Wind 파라미터 설정
w_hor = 1           # 수평풍 속도 스케일  [m/s]
scale = 0.1         # Perlin noise scale
seed = 1            # seed number

# 3차원 quiver의 길이 파라미터 (map 크기에 따라 보기 좋게 수정해주는 역할)
# arrow_size = 30

######################################################################################################
# Plot 1
######################################################################################################

X, Y, Z = np.meshgrid(np.linspace(x_min, x_max, num_x), np.linspace(y_min, y_max, num_y), altitudes)

# Horizontal Wind Vector Field 
u, v = generate_wind_field(num_x, num_y, scale, w_hor, seed)

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
w = np.empty((num_x, num_y, num_z))


xt_old = xc
yt_old = yc

for i, z in enumerate(altitudes):
    
    w_vert, xt, yt = allen_model(resolution,z, zi, w_star, x_min, x_max, y_min, y_max, u, v, xc, yc, xt_old, yt_old)
    if w_vert.shape == (num_x, num_y):
        w[:,:,i] = w_vert
    else:
        raise ValueError(f"Shape mismatch: w_vert.shape = {w_vert.shape}, expected = {(num_x, num_y)}")

    xt_old = xt
    yt_old = yt
    if i%50 ==0:
        print("Writing... please wait")

t.toc()
np.save(f'C:/Users/seung/WindData/u_{ResolutionType}', u)
np.save(f'C:/Users/seung/WindData/v_{ResolutionType}', v)
np.save(f'C:/Users/seung/WindData/w_{ResolutionType}', w)
'''
# OneDrive 경로 설정
onedrive_path = 'C:/Users/seung/OneDrive/문서/WindData/'

# 파일 저장 경로 설정
u_file_path = f'{onedrive_path}u_coarse.npy'
v_file_path = f'{onedrive_path}v_coarse.npy'
w_file_path = f'{onedrive_path}w_coarse.npy'

# 데이터 저장
np.save(u_file_path, data)
np.save(v_file_path, data)
np.save(w_file_path, data)

print("파일이 OneDrive에 저장되었습니다.")
'''      
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
