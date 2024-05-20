import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Allen_Ben():
    """
    Wz = function (z : altitude) : calculating updraft's velocity from altitudes using improved Allen model considered Bencatel model
    
    Args:
        z: 고도 (m)
        zi: 대류 혼합층 두께 (m)
        w_star: 대류 속도 스케일 (m/s)
        Wx: x 방향 수평 바람 속도 (m/s)
        Wy: y 방향 수평 바람 속도 (m/s)
        xc: 지표면에서 상승기류 중심의 x 좌표 (m)
        yc: 지표면에서 상승기류 중심의 y 좌표 (m)
        u: 지표면에서 상승기류 중심의 x 방향 이동 속도 (m/s)
        v: 지표면에서 상승기류 중심의 y 방향 이동 속도 (m/s)
        r1_ratio: 상승기류 코어 반지름 비율 (r1 / r2)

    Returns:
        X: x 좌표 배열 (m)
        Y: y 좌표 배열 (m)
        w_total: 총 상승기류 속도 배열 (m/s)
        
    """
    def __init__(self, zi = 1000, w_star = 2, r1_ratio = 0.80, S = 300 **2,Wx= 1, Wy = 0.5, xc=0, yc=0, u=0, v=0):
        '''
        # 매개변수 설정
        zi = 1000           # 대류 혼합층 두께 (m)
        w_star = 2          # 대류 속도 스케일 (m/s)
        Wx = 1              # x 방향 수평 바람 속도 (m/s)
        Wy = 0.5            # y 방향 수평 바람 속도 (m/s)
        r1_ratio = 0.80     # r1/r2 비율
        area = 300**2       # 해당 영역 넓이
        '''
        self.zi = zi
        self.w_star = w_star
        self.r1_ratio = r1_ratio
        self.S = S
        self.Wx = Wx
        self.Wy = Wy
        self.xc = xc
        self.yc = yc
        self.u = u
        self.v = v
        
        
    #def allen_model_with_bencatel(z, zi, w_star, r1_ratio, S, Wx, Wy, xc=0, yc=0, u=0, v=0):
    def estimate(self,z):

        # 상승기류 외곽 반지름 계산
        # r2 = max(10, 0.102 * (z / zi)**(1/3) * (1 - 0.25 * z / zi)) * zi
        r2 = (0.102 * (z / self.zi)**(1/3) * (1 - 0.25 * z / self.zi)) * self.zi

        # 평균 상승기류 속도 계산 (Lenschow 방정식)
        w_avg = 1.0 * (z / self.zi)**(1/3) * (1 - 1.1 * z / self.zi) * self.w_star

        # 상승기류 코어 반지름 계산
        r1 = self.r1_ratio * r2

        # 상승기류 최대 속도 계산
        w_peak = 3 * w_avg * (r2**3 - r1**3) / (2 * r1 * (r2**2 - r1**2))

        # 벨 모양 곡선 형상 상수
        if self.r1_ratio == 0.14:
            K = [1.5352, 2.5826, -0.0113, 0.0008]
        elif self.r1_ratio == 0.25:
            K = [1.5265, 3.6054, -0.0176, 0.0005]
        elif self.r1_ratio == 0.36:
            K = [1.4866, 4.8354, -0.0320, 0.0001]
        elif self.r1_ratio == 0.47:
            K = [1.2042, 7.7904,  0.0848, 0.0001]
        elif self.r1_ratio == 0.58:
            K = [0.8816, 13.972,  0.3404, 0.0001]
        elif self.r1_ratio == 0.69:
            K = [0.7067, 23.994,  0.5689, 0.0002]
        elif self.r1_ratio == 0.80:
            K = [0.6189, 42.797,  0.7157, 0.0001]
        else: KeyError("Please check r1/r2.")

        k1 = K[0]
        k2 = K[1]
        k3 = K[2]
        k4 = K[3]

        # Bencatel 모델 적용: 상승기류 중심선 위치 계산
        xt = self.xc + (self.Wx - self.u) * z / w_avg
        yt = self.yc + (self.Wy - self.v) * z / w_avg

        # 2D 좌표 배열 생성
        x = np.linspace(xt - 2 * r2, xt + 2 * r2, 100)
        y = np.linspace(yt - 2 * r2, yt + 2 * r2, 100)
        X, Y = np.meshgrid(x, y)

        # 상승기류 중심에서의 거리 계산
        R = np.sqrt((X - xt)**2 + (Y - yt)**2)

        # 상승기류 속도 계산
        w = w_peak * ((1 / (1 + abs(k1 * R / r2 + k3)**k2)) + k4 * (R / r2))    

        # 환경 하강 속도 계산
        w_e = -w_peak * np.pi * r2**2 / (2 * self.S - np.pi * r2**2)  # 300x300 m^2 영역 가정

        # 총 상승기류 속도 계산
        w_total = w * (1 - w_e / w_peak) + w_e

        return X, Y, R, w_total
##########################################################################################
# 1. 특정 고도에서의 3D Surface Plot
z = 800  # 고도 (m)
ABen = Allen_Ben()
# Bencatel 모델을 고려한 Allen 모델 계산
X, Y, R, w_total = ABen.estimate(z)

# 3D 서피스 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, w_total, cmap='jet')
fig.colorbar(surf, label="Updraft Velocity (m/s)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Updraft Velocity (m/s)")
plt.title("Allen Model with Bencatel (z = 800 m)")
plt.show()


############################################################################################
# 여러 고도에서 상승기류 속도 계산
altitudes = [200, 400, 600, 800]  # 고도 (m)
for z in altitudes:
    x, y, r, w_total = ABen.estimate(z)
    plt.plot(r, w_total, label=f"Altitude = {z} m")

# 그래프 설정
plt.xlabel("Distance from Thermal Center (m)")
plt.ylabel("Updraft Velocity (m/s)")
plt.title("Allen Updraft Model")
plt.grid(True)
plt.show()
