import numpy as np
import matplotlib.pyplot as plt
from bit_Thermal import BITStar
from rrt_star_Thermal import RRTStar
def compare_algorithms(x_start, x_goal, wind_data, iter_max=1000, va=20, step_len=300, eta=40):
    u, v, w = wind_data

    # Initialize both algorithms
    bit = BITStar(x_start, x_goal, eta, iter_max, va, u, v, w)
    rrt = RRTStar(x_start, x_goal, r=500, step_len=step_len, iter_max=iter_max, va=va, u=u, v=v, w=w)

    # Run planning for both algorithms
    bit_path = bit.planning()
    rrt_path = rrt.planning()

    # Calculate the costs of both paths
    bit_cost = bit.calculate_cost()
    rrt_cost = rrt.calculate_cost()

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
    wind_data = (np.zeros((100, 100, 100)), np.zeros((100, 100, 100)), np.zeros((100, 100, 100)))  # Placeholder wind data

    # Compare the two algorithms
    compare_algorithms(x_start, x_goal, wind_data)


if __name__ == "__main__":
    main()
