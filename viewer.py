"""
FINAL RESULT VIEWER (PATH, WIND, OBSTACLES)
@author : Minjo Jung
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

class VIEWER:
    def __init__(self, planner, map):
        
        zi = 4000
        arrow_size = 30
        plot_uv = False      # True : u,v,w / False : w만

        num_th = 3
        ResolutionType = 'coarse'
        resolution = 200

        self.x_range = (-1000, 5000)
        self.y_range = (-1000, 5000)
        self.z_range = (0, 4000)

        self.fig = plt.figure(figsize=(15,12))

        self.ax = self.fig.add_subplot(111,projection = '3d')
        self.ax.set_xlim([self.x_range[0], self.x_range[1]])
        self.ax.set_ylim([self.y_range[0], self.y_range[1]])
        self.ax.set_zlim([self.z_range[0], self.z_range[1]])

        windData_path = 'C:/Users/LiCS/Documents/MJ/KAIST/Paper/2025 ACC/Code/windData/'        #Windows
        
        self.u = np.load(f'{windData_path}u_{ResolutionType}_th{num_th}.npy')
        self.v = np.load(f'{windData_path}v_{ResolutionType}_th{num_th}.npy')
        self.w = np.load(f'{windData_path}w_{ResolutionType}_th{num_th}.npy')
        
        x_len = self.x_range[1] - self.x_range[0]
        y_len = self.y_range[1] - self.y_range[0]

        num_x = int((x_len / resolution) + 1)
        num_y = int((y_len / resolution) + 1)
        num_z = int(((zi - resolution) / resolution) + 1)
        
        altitudes = np.linspace(resolution, zi, num_z)
        X, Y, Z = np.meshgrid(np.linspace(self.x_range[0], self.x_range[1], num_x), np.linspace(self.y_range[0], self.y_range[1], num_y), altitudes)

        # Wind speed [m/s]
        s = np.zeros([num_x, num_y, num_z])
        for k in range(num_z):
            s[:,:,k] = np.sqrt(self.u**2 + self.v**2 + self.w[:,:,k]**2)

        # Figure and Plot Settings
        norm = Normalize(vmin=np.min(s), vmax=np.max(s))
        cmap = cm.winter.reversed()

        X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
        # print('-----')
        # print(f'X_flat : {X_flat.shape}, Y_flat : {Y_flat.shape}, Z_flat : {Z_flat.shape}')
        u_flat_, v_flat_, w_flat = self.u.flatten(), self.v.flatten(), self.w.flatten()
        u_flat = u_flat_
        v_flat = v_flat_
        for k in range(num_z-1):
            u_flat = np.concatenate((u_flat, u_flat_), axis = 0)
            v_flat = np.concatenate((v_flat, v_flat_), axis = 0)
        # print(f'u_flat : {u_flat.shape}, v_flat : {v_flat.shape}, w_flat : {w_flat.shape}')
        s_flat = s.flatten()

        # Calculate colors for each arrow
        colors = cmap(norm(s_flat))

        # Plot each quiver individually with corresponding color
        for i in range(len(X_flat)):
            if plot_uv: # 1. u,v,w 방향 모두
                self.ax.quiver(X_flat[i], Y_flat[i], Z_flat[i], u_flat[i], v_flat[i], w_flat[i], color=colors[i], length=arrow_size)
            else:       # 2. w 방향만
                self.ax.quiver(X_flat[i], Y_flat[i], Z_flat[i], 0, 0, w_flat[i], color=colors[i], length=arrow_size)  

        # Create color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(s_flat)
        self.fig.colorbar(sm, ax=self.ax, shrink=0.5, aspect=5)

        plt.plot(planner.x_start.x, planner.x_start.y, planner.x_start.z, "bs", linewidth=3)
        plt.plot(planner.x_goal.x, planner.x_goal.y, planner.x_goal.z, "rx", linewidth=3)

        # Plot the path
        if len(planner.path_x) is not None:
            self.ax.plot(planner.path_x, planner.path_y, planner.path_z , linewidth=2, color='r',linestyle ='--')

        # Plot the obstacles
        if len(map.obs) is not None:
            for obs in map.obs:
                obs.draw(self.ax)
        
        # Set title and labels
        plt.title('Wind Vector Map (coarse=200)')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')

        # Show the plot
        plt.show()
