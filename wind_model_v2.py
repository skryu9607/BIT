import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from noise import pnoise2

"""
Code 정보
1. 수평풍 : Perlin Noise 를 이용 (wind_model_v1 과의 차이점)
2. 수직풍 : leaning 을 고려한 Gedeon 모델
3. 수직풍을 고려할 때, 수평풍 Wx, Wy를 수정 (사전 정의해야 하는 파라미터 -> 상승기류 내부 영역에 존재하는 수평풍 u,v 의 평균)
"""

"""
Plot 정보
1. 특정 고도에서의 2차원 horiziontal wind 분포 (수평 x-y 방향만 고려)
2. 특정 고도에서의 thermal updraft 속도 분포 (수직 z 방향만 고려)
3. 수평 및 수직 방향 바람장 plots
"""

######################################################################################################

# Perlin 노이즈를 사용하여 2D wind field map 생성
class WINDS: # 2D wind 
    def __init__(self,seed,wind_speed):
        self.ws = wind_speed
        self.seed = seed
    def ambient_winds(self,num_x,num_y,num_z,scale = 0.1):
        np.random.seed(self.seed)

        u = np.zeros((num_x, num_y, num_z))
        v = np.zeros((num_x, num_y, num_z))
        
        for i in range(num_x):
            for j in range(num_y):
                
                noise1 = pnoise2(i * scale, j * scale, octaves=1, base=self.seed)
                noise2 = pnoise2(i * scale, j * scale, octaves=1, base=self.seed + 10)

                angle     = noise1 * 2 * np.pi
                magnitude = noise2 * self.ws + self.ws

                # 고도 (k) 에 따른 수평풍 변화는 없음
                for k in range(num_z):    
                    u[i, j, k] = magnitude * np.cos(angle)
                    v[i, j, k] = magnitude * np.sin(angle)
        
        return u, v
class Thermals(WINDS):
    '''
    Inputs : Thermals의 위치 xc,yc, 
    
    '''
    def __init__(self,zi,w_star,xc,yc):
        self.zi = zi
        self.w_star = w_star
        self.xc = xc
        self.yc = yc
        self.num_x = 30
        self.num_y = 30
        self.num_z = 5
        self.altitudes = np.linspace(100, 0.9*zi, self.num_z)
        w_hor = 1
        self.X, self.Y, self.Z = np.meshgrid(np.linspace(-map_x, map_x, self.num_x), np.linspace(-map_y, map_y, self.num_y), self.altitudes)
        # Horizontal Wind Vector Field 
        self.u, self.v = WINDS.ambient_winds(self,scale, w_hor)
    # Thermal Model Function
    def lenschow_model_with_gedeon(self, z, map_x, map_y, u0=0, v0=0):
        
        # 상승기류 외곽 반지름 계산 [m]
        d = self.zi * (0.16 * (z / self.zi)**(1/3)) * (1 - (0.25 * z)/self.zi)

        # 평균 상승기류 속도 계산 (Lenschow 방정식) [m/s]
        w_core = 1.0 * (z / self.zi)**(1/3) * (1 - 1.1 * z / self.zi) * self.w_star    
        
        # 그리드 간격
        grid_x = 2 * map_x / self.num_x
        grid_y = 2 * map_y / self.num_y
        
        # (d/2) 가 각 축에 대해 몇 개의 grid로 이루어지는지
        num_grid_x = int((d/2) // grid_x)
        num_grid_y = int((d/2) // grid_y)
        # thermal 중심 (xc,yc)이 맵 중심 (0,0)에 위치할 때, 각 축의 인덱스 상/하한 (thermal 내부에 존재하는 수평풍에 대해) 
        idx_x_low = int(self.num_x / 2 - 1 - num_grid_x)
        idx_x_upp = int(self.num_x / 2 - 1 + num_grid_x)
        
        idx_y_low = int(self.num_y / 2 - 1 - num_grid_y)
        idx_y_upp = int(self.num_y / 2 - 1 + num_grid_y)
        
        # Wx, Wy 는 해당 고도의 상승기류 내부 영역에 존재하는 수평풍 (u,v) 의 평균!!
        Wx = 0
        Wy = 0
        
        for idx_x in range(idx_x_low, idx_x_upp):
            for idx_y in range(idx_y_low, idx_y_upp):
                Wx = Wx + self.u[idx_x,idx_y,i]
                Wy = Wy + self.v[idx_x,idx_y,i]
        
        Wx = Wx / ((2*num_grid_x + 1)*(2*num_grid_y + 1))
        Wy = Wy / ((2*num_grid_x + 1)*(2*num_grid_y + 1))

        # Bencatel 모델 적용: 상승기류 중심선 위치 계산 [m]
        xt = self.xc + (Wx - u0) * z / w_core
        yt = self.yc + (Wy - v0) * z / w_core

        # 2D 좌표 배열 생성
        x = np.linspace(-map_x, map_x, num_x)
        y = np.linspace(-map_y, map_y, num_y)
        X, Y = np.meshgrid(x, y)

        # 상승기류 중심에서의 거리 계산 [m]
        r = np.sqrt((X - xt)**2 + (Y - yt)**2)

        # 상승기류 속도 [m/s]
        w = w_core * np.exp(- (r/(d/2))**2) * (1 - (r/(d/2))**2)

        return w
    
    def update(self):# Vertical Wind (Thermal Updraft)
        w = np.empty((self.num_x, self.num_y, self.num_z))

        for i, z in enumerate(self.altitudes):
            
            w_vert = self.lenschow_model_with_gedeon(self,z, zi, w_star, map_x, map_y, num_x, num_y)
            w[:,:,i] = w_vert
            
    def plot(self):
        # 3D Surface Plot in a specific altitude 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(self.X[:,:,round(self.num_z/2)], self.Y[:,:,round(self.num_z/2)], self.w[:,:,round(self.num_z/2)], cmap=cm.winter.reversed())
        fig.colorbar(surf, label="Updraft Velocity (m/s)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Updraft Velocity (m/s)")
        plt.title("Updraft Thermal Model with Leaning")
        plt.show()
        # Wind speed [m/s]
        s = np.sqrt(u**2 + v**2 + w**2)

        ######################################################################################################

        # Figure and Plot Settings

        norm = Normalize(vmin=np.min(s), vmax=np.max(s))
        cmap = cm.winter.reversed()

        X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
        u_flat, v_flat, w_flat = u.flatten(), v.flatten(), w.flatten()
        s_flat = s.flatten()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Calculate colors for each arrow
        colors = cmap(norm(s_flat))

        # Plot each quiver individually with corresponding color
        for i in range(len(X_flat)):
            # ax.quiver(X_flat[i], Y_flat[i], Z_flat[i], u_flat[i], v_flat[i], w_flat[i], color=colors[i], length=50, normalize=True)
            ax.quiver(X_flat[i], Y_flat[i], Z_flat[i], u_flat[i], v_flat[i], w_flat[i], color=colors[i], length=arrow_size)

        # Create color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(s_flat)
        fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)

        # Set title and labels
        plt.title('Wind Vector Map')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')

        # Show the plot
        plt.show()
    ######################################################################################################
######################################################################################################
# Thermal 파라미터 설정
zi = 1000           # 대류 혼합층 두께 [m]
w_star = 10         # 상승기류 속도 스케일 [m/s]
xc = 0              # 상승기류 중심의 x좌표
yc = 0              # 상승기류 중심의 y좌표

# 맵 설정 [m] : x, y 축 절반 길이 (맵 중심은 (0,0))
map_x = 5000
map_y = 4000


# Horizontal Wind 파라미터 설정
w_hor = 1         # 수평풍 속도 스케일  [m/s]
scale = 0.1         # Perlin noise scale
# 3차원 quiver의 길이 파라미터 (map 크기에 따라 보기 좋게 수정해주는 역할)
arrow_size = 30

######################################################################################################
'''
# Wind speed [m/s]
s = np.sqrt(u**2 + v**2 + w**2)

######################################################################################################

# Figure and Plot Settings

norm = Normalize(vmin=np.min(s), vmax=np.max(s))
cmap = cm.winter.reversed()

X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
u_flat, v_flat, w_flat = u.flatten(), v.flatten(), w.flatten()
s_flat = s.flatten()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Calculate colors for each arrow
colors = cmap(norm(s_flat))

# Plot each quiver individually with corresponding color
for i in range(len(X_flat)):
    # ax.quiver(X_flat[i], Y_flat[i], Z_flat[i], u_flat[i], v_flat[i], w_flat[i], color=colors[i], length=50, normalize=True)
    ax.quiver(X_flat[i], Y_flat[i], Z_flat[i], u_flat[i], v_flat[i], w_flat[i], color=colors[i], length=arrow_size)

# Create color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(s_flat)
fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)

# Set title and labels
plt.title('Wind Vector Map')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')

# Show the plot
plt.show()
'''
