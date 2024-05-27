import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm

"""
Plot 정보
1. 특정 고도에서의 2차원 horiziontal wind 분포 (수평 x-y 방향만 고려)
2. 특정 고도에서의 thermal updraft 속도 분포 (수직 z 방향만 고려)
3. 수평 및 수직 방향 바람장 plots

developed by Minjo Jung, 2024.05.26
"""

######################################################################################################

# Thermal Model Function

def lenschow_model_with_gedeon(z, zi, w_star, Wx, Wy, map_x, map_y, num_x, num_y, xc=0, yc=0, u=0, v=0):
    
    # 상승기류 외곽 반지름 계산 [m]
    d = zi * (0.16 * (z / zi)**(1/3)) * (1 - (0.25 * z)/zi)

    # 평균 상승기류 속도 계산 (Lenschow 방정식) [m/s]
    w_core = 1.0 * (z / zi)**(1/3) * (1 - 1.1 * z / zi) * w_star    

    # Bencatel 모델 적용: 상승기류 중심선 위치 계산 [m]
    xt = xc + (Wx - u) * z / w_core
    yt = yc + (Wy - v) * z / w_core

    # 2D 좌표 배열 생성
    x = np.linspace(-map_x, map_x, num_x)
    y = np.linspace(-map_y, map_y, num_y)
    X, Y = np.meshgrid(x, y)

    # 상승기류 중심에서의 거리 계산 [m]
    r = np.sqrt((X - xt)**2 + (Y - yt)**2)

    # 상승기류 속도 [m/s]
    w = w_core * np.exp(- (r/(d/2))**2) * (1 -(r/(d/2))**2)

    return w

######################################################################################################

# Thermal 파라미터 설정
zi = 1000           # 대류 혼합층 두께 [m]
w_star = 10         # 상승기류 속도 스케일 [m/s]
Wx = 0.1            # x 방향 수평 바람 속도 [m/s] -> Thermal 모델의 leaning 을 구현하기 위해 사용
Wy = 0.1            # y 방향 수평 바람 속도 [m/s] -> Thermal 모델의 leaning 을 구현하기 위해 사용
w_hor = 1           # 수평풍 속도 스케일 [m/s]

# 맵 설정 [m] : x, y 축 절반 길이 (맵 중심은 (0,0))
map_x = 500
map_y = 400

# Resolution 설정 (개수)
num_x = 30
num_y = 30
num_z = 5

# 3차원 quiver의 길이 파라미터 (map 크기에 따라 보기 좋게 수정해주는 역할)
arrow_size = 30

altitudes = np.linspace(100, 0.9*zi, num_z)

######################################################################################################
 
X, Y, Z = np.meshgrid(np.linspace(-map_x, map_x, num_x), np.linspace(-map_y, map_y, num_y), altitudes)

# Horizontal Wind Vector Field Definition
u = w_hor * (np.sin(np.pi * X))
v = w_hor * (np.cos(np.pi * Y))

# Horizontal Wind Plot
ax = plt.axes()

wind = ax.quiver(X[:,:,0], Y[:,:,0], u[:,:,0], v[:,:,0], np.sqrt(u[:,:,0]**2 + v[:,:,0]**2), cmap=cm.winter.reversed())
plt.colorbar(wind, ax=ax, orientation='vertical').set_label('Wind Speed (m/s)')

# Set title and labels
plt.title('Wind Vector Map (2D)')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')

# Show the plot
plt.show()

######################################################################################################

# Vertical Wind (Thermal Updraft)
w = np.empty((num_x, num_y, num_z))

for i, z in enumerate(altitudes):
    w_vert = lenschow_model_with_gedeon(z, zi, w_star, Wx, Wy, map_x, map_y, num_x, num_y)
    w[:,:,i] = w_vert

# 3D Surface Plot in a specific altitude 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X[:,:,round(num_z/2)], Y[:,:,round(num_z/2)], w[:,:,round(num_z/2)], cmap=cm.winter.reversed())
fig.colorbar(surf, label="Updraft Velocity (m/s)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Updraft Velocity (m/s)")
plt.title("Updraft Thermal Model with Leaning")
plt.show()

######################################################################################################

# Wind speed [m/s]
s = np.sqrt(u**2 + v**2 + w**2)
print(np.max(s))

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
