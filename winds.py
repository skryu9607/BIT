"""
Thermal updrafts modeling
@author : Minjo Jung
"""

import numpy as np

def wind_catcher(x,y,z,u,v,w):

    # 맵 data [m] : x, y 축 절반 길이 (맵 중심은 (0,0))
    # x : -map_x ~ map_x
    # y : -map_x ~ map_x
    # z : resolution ~ 0.9*zi

    x_min = -1000.
    x_max = 5000.
    y_min = -1000.
    y_max = 5000.
    zi = 4000.
    resolution = 10.
    '''
    if x > x_max or x < x_min or y > y_max or y < y_min or z < resolution or z > 0.9*zi:
        raise ValueError
    '''
    if x > x_max or x < x_min or y > y_max or y < y_min:
        raise ValueError
    if z < resolution:
        idx_z = round(0)
    else:
        idx_z = round((z - resolution) / resolution)
    idx_x = round((x - x_min) / resolution)
    idx_y = round((y - y_min) / resolution)

    # u, v 가 2차원 데이터
    if z > 0.9 * zi:
        wind = [u[idx_x,idx_y], v[idx_x,idx_y], 0.]
    else:
        wind = [u[idx_x,idx_y], v[idx_x,idx_y], w[idx_x,idx_y,idx_z]]

    return wind

'''Windows '''
u = np.load('C:/Users/seung/WindData/u.npy')
v = np.load('C:/Users/seung/WindData/v.npy')
w = np.load('C:/Users/seung/WindData/w.npy')

wind = wind_catcher(-1000,5000,3600,u,v,w)
print(wind)
