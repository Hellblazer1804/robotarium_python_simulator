import rps.robotarium as robotarium
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Number of robots
N = 10
L = completeGL(N)

# Define initial conditions
initial_conditions = np.asarray([[1.25, 0.25, 0],[1, 0.5, 0],[1, -0.5, 0],
                                 [-1, -0.75, 0],[0.1, 0.2, 0],[0.2, -0.6, 0],
                                 [-0.75, -0.1, 0],[-1, 0, 0],[-0.8, -0.25, 0],
                                 [1.3, -0.4, 0]]).T

# Create Robotarium object
r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=True,
    sim_in_real_time=False,
    initial_conditions=initial_conditions
)

# Simulation parameters
iterations = 1000
MIN_ITERATIONS = 1000  # Minimum iterations before convergence check
CONVERGENCE_THRESHOLD = 1e-3  # Looser convergence threshold

# Dynamics mapping
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Workspace parameters
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.0, 1.0
res = 0.05
convergence = 0

# Generate grid points
X, Y = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))
grid_points = np.vstack([X.ravel(), Y.ravel()])  # (2, M)

# Robot parameters (heterogeneous)
sensor_types = {1: [0,1,2,3,4],2:[5,6,7,8,9]}  # Sensor type assignments
importance = {
    1: {0: 1, 1: 1, 2: 1,3:1,4:1},  # Importance for sensor type 1 (robots 0, 1, 2)
    2: {5: 1, 6: 1, 7: 1,8:1,9:1}  # Importance for sensor type 2 (robots 2, 3, 4)
}

# Density function (constant for this example)
phi_q = np.ones(grid_points.shape[1])  # Uniform density


def plot_robot_trajectories(robot_trajectories, initial_positions, final_positions, scenario_name):
    """
    Plots the robot trajectories along with their final Voronoi partitions.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Convert trajectory lists to arrays
    for i in range(N):
        trajectory = np.array(robot_trajectories[i])  # Convert list to numpy array

        # Mark initial and final positions
        ax.scatter(initial_positions[i, 0], initial_positions[i, 1], c='black', marker='o', s=50,
                   label="Start" if i == 0 else "")
        ax.scatter(final_positions[i, 0], final_positions[i, 1], c='red', marker='x', s=50,
                   label="End" if i == 0 else "")

        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], label=f"Robot {i}", linewidth=2)

    # Compute Voronoi partition for final positions
    vor = Voronoi(final_positions)

    # Plot Voronoi partitions
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=1.5, alpha=0.6)

    # Configure plot
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(f"Robot Trajectories for Type Coverage: {scenario_name}", fontsize=16)
    plt.xlabel("X Position", fontsize=12, fontweight='bold')
    plt.ylabel("Y Position", fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.10, 1), borderaxespad=0.)
    plt.grid()

    # Save and show plot
    plt.savefig(f"data/type_trajectories_case_10.png")


# Control gain
k_gain = 1.0

# Cost history and convergence tracking
cost_history = []
robot_trajectories = {i: [] for i in range(N)}  # Store (x, y) positions over time
wij = []
pose = []
cwi = []
dxi = []
ui = []
dist_robots = np.zeros(N)
cumulative_dist = []

# Main simulation loop
for k in range(iterations):
    x = r.get_poses()
    x_si = uni_to_si_states(x)[:2, :]  # (2, N)

    # Store position history
    for i in range(N):
        robot_trajectories[i].append(x_si[:, i].copy())

    c_v = np.zeros((N, 2))  # centroid vector
    w_v = np.zeros(N)  # weight vector
    locational_cost = 0

    # Process each point in the grid
    for ix in np.arange(x_min, x_max, res):
        for iy in np.arange(y_min, y_max, res):
            for sensor_type, robots in sensor_types.items():
                distances = np.full(N, np.inf)

                # Calculate distances for robots with this sensor type
                for robot in robots:
                    distances[robot] = np.sqrt((ix - x_si[0, robot]) ** 2 + (iy - x_si[1, robot]) ** 2)

                # Find the closest robot within the current sensor type
                min_index = robots[np.argmin([distances[r] for r in robots])]

                # Get importance value for the closest robot
                importance_value = importance[sensor_type].get(min_index, 1.0)

                # Update centroid calculation
                c_v[min_index][0] += ix * importance_value
                c_v[min_index][1] += iy * importance_value
                w_v[min_index] += importance_value

                # Update cost
                locational_cost += distances[min_index] * importance_value * (res ** 2)

    wij.append(w_v)

    # Initialize the single-integrator control inputs
    si_velocities = np.zeros((2, N))

    for i in range(N):
        if w_v[i] > 0:
            c_v[i] /= w_v[i]
        else:
            c_v[i] = x_si[:, i]  # Fallback to current position

        # Calculate desired velocity
        si_velocities[:, i] = k_gain * (c_v[i] - x_si[:, i])

    # Check convergence
    if len(cost_history) > 1 and abs(locational_cost - cost_history[-1]) < 0.0001 and convergence == 0:
        convergence = k

    cost_history.append(locational_cost)

    # Use the barrier certificate to avoid collisions (optional)
    # si_velocities = si_barrier_cert(si_velocities, x_si)

    # Transform single integrator to unicycle
    dxu = si_to_uni_dyn(si_velocities, x)

    # Set the velocities of agents 1,...,N
    r.set_velocities(np.arange(N), dxu)

    # Iterate the simulation
    r.step()

# Save results and cleanup
final_positions = np.array([robot_trajectories[i][-1] for i in range(N)])  # Last recorded position
initial_positions = np.array([robot_trajectories[i][0] for i in range(N)])  # First recorded position

plot_robot_trajectories(robot_trajectories, initial_positions, final_positions, scenario_name="Type Coverage")

# Save the data to csv file
df = pd.DataFrame(cost_history)
df.to_csv('data/type_coverage_cost_case_10.csv', index=False)

with open("data/type_coverage_case_10.txt", "w") as f:
    f.write(str(convergence))

# Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
