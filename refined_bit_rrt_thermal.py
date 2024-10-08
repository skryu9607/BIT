import numpy as np
import matplotlib.pyplot as plt
from bit_Thermal import BITStar
from rrt_star_Thermal import RRTStar
def compare_algorithms(x_start, x_goal, ResolutionType, iter_max=1000, va=20, step_len=300, eta=40):
    #ResolutionType = 'normal'
    
    # Wind Data Path : Seung
    onedrive_path = '/Users/seung/WindData/'
    # Wind Data Path : MinJo
    #onedrive_path = 'C:/Users/LiCS/Documents/MJ/KAIST/Paper/2025 ACC/Code/windData/'
    
    #Mac : OneDrive
    u = np.load(f'{onedrive_path}/{ResolutionType}/u_{ResolutionType}.npy')
    v = np.load(f'{onedrive_path}/{ResolutionType}/v_{ResolutionType}.npy')
    w = np.load(f'{onedrive_path}/{ResolutionType}/w_{ResolutionType}.npy')

    eta = 2 * 1 * 20
    iter_max = 400
    va = 20
    # Initialize both algorithms
    rrt = RRTStar(x_start, x_goal, r=500, step_len=step_len, iter_max=iter_max, va=va, u=u, v=v, w=w)
    bit = BITStar(x_start, x_goal, eta, iter_max, va, u, v, w)
    bit_path = bit.planning()
    rrt_path = rrt.planning()
    # Run planning for both algorithms

    # Calculate the costs of both paths
    bit_cost = bit.g_T[x_goal]
    rrt_cost = rrt.g_T[x_goal]

    # Print paths and costs
    print("BIT* Path:", bit_path)
    print("RRT* Path:", rrt_path)
    print(f"BIT* Cost: {bit_cost}, RRT* Cost: {rrt_cost}")

    if bit_cost < rrt_cost:
        print("BIT* is more efficient.")
    else:
        print("RRT* is more efficient or equal.")


def main():
    x_start = (0.0, 0.0, 10.0)  # Starting node
    x_goal = (3000, 3000, 3000)  # Goal node

    # Compare the two algorithms
    compare_algorithms(x_start, x_goal, 'normal')


if __name__ == "__main__":
    main()
