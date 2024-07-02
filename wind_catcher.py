import numpy as np

def wind_catcher(x,y,z,u,v,w):

    # 맵 data [m] : x, y 축 절반 길이 (맵 중심은 (0,0))
    # x : -map_x ~ map_x
    # y : -map_x ~ map_x
    # z : resolution ~ 0.9*zi

    map_x = 500
    map_y = 400
    resolution = 1
    
    idx_x = int((x + map_x) / resolution)
    idx_y = int((y + map_y) / resolution)
    idx_z = int((z - resolution) / resolution)

    wind = [u[idx_x,idx_y], v[idx_x,idx_y], w[idx_x,idx_y,idx_z]]

    return wind


u = np.load('C:/Users/LiCS/Documents/MJ/KAIST/Paper/2025 ACC/Wind Energy/Code/Thermal/u.npy')
v = np.load('C:/Users/LiCS/Documents/MJ/KAIST/Paper/2025 ACC/Wind Energy/Code/Thermal/v.npy')
w = np.load('C:/Users/LiCS/Documents/MJ/KAIST/Paper/2025 ACC/Wind Energy/Code/Thermal/w.npy')

print(type(u[0,0]))
# print(u)
# print(u.shape)
# print(u[0,0,50])

# print(v)
# print(v.shape)
# print(v[0,0,50])

# print(w)
# print(w.shape)
# print(w[0,0,50])

wind = wind_catcher(-500,-400,900,u,v,w)
print(wind)