import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
from rps.utilities.graph import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d 

# Number of robots
N = 10
L = completeGL(N)

# Define initial conditions
initial_conditions = np.asarray([[1.25, 0.25, 0],[1, 0.5, 0],[1, -0.5, 0],[-1, -0.75, 0],[0.1, 0.2, 0],[0.2, -0.6, 0],[-0.75, -0.1, 0],[-1, 0, 0],[-0.8, -0.25, 0],[1.3, -0.4, 0]]).T

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
Vr = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0])  # Max velocities (heterogeneous speeds)

# Density function (constant for this example)
phi_q = np.ones(grid_points.shape[1])  # Uniform density

# Multiplicatively weighted Voronoi partitioning
def weighted_voronoi_partition(x_si, v_i):
    """Compute the multiplicatively weighted Voronoi partition."""
    D_weighted = np.zeros((grid_points.shape[1], N))
    for i in range(N):
        dist = np.linalg.norm(grid_points - x_si[:, i, None], axis=0)
        D_weighted[:, i] = dist / v_i[i]

    return np.argmin(D_weighted, axis=1)

def plot_robot_trajectories(robot_trajectories, initial_positions, final_positions, scenario_name):
    """
    Plots the robot trajectories along with their final Voronoi partitions.

    Parameters:
        robot_trajectories: dict
            Dictionary mapping robot index to a list of (x, y) positions.
        initial_positions: np.ndarray
            Array of initial robot positions.
        final_positions: np.ndarray
            Array of final robot positions.
        scenario_name: str
            Name of the scenario for labeling the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Convert trajectory lists to arrays
    for i in range(N):
        trajectory = np.array(robot_trajectories[i])  # Convert list to numpy array
         # Mark initial and final positions
        ax.scatter(initial_positions[i, 0], initial_positions[i, 1], c='black', marker='o', s=50, label="Start" if i == 0 else "")
        ax.scatter(final_positions[i, 0], final_positions[i, 1], c='red', marker='x', s=50, label="End" if i == 0 else "")
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], label=f"Robot {i}", linewidth=2)

    # Compute Voronoi partition for final positions
    vor = Voronoi(final_positions)

    # Plot Voronoi partitions
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=1.5, alpha=0.6)

    # Configure plot
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(f"Robot Trajectories for Mobility Coverage: {scenario_name}", fontsize=16)
    plt.xlabel("X Position", fontsize=12, fontweight='bold')
    plt.ylabel("Y Position", fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.10, 1), borderaxespad=0.)
    plt.grid()
    
    # Save and show plot
    plt.savefig(f"mobility_trajectories_case_10.png")

# Control gain
k_gain = 1.0

# Cost history and convergence tracking
cost_history = []
robot_trajectories = {i: [] for i in range(N)}  # Store (x, y) positions over time

# Main simulation loop
for k in range(iterations):
    x = r.get_poses()
    x_si = uni_to_si_states(x)[:2, :]  # (2, N)

    # Compute multiplicatively weighted Voronoi partition
    robot_assignments = weighted_voronoi_partition(x_si, Vr)

    # Initialize cost components
    covered_cost = 0.0

    # Calculate covered cost and centroids
    c_v = np.zeros((N, 2))
    w_v = np.zeros(N)

    for i in range(N):
        # Points assigned to this robot
        robot_trajectories[i].append(x_si[:, i].copy())  # Store position history
        valid_points = (robot_assignments == i)

        if np.any(valid_points):
            # Get distances and weights for valid points
            dist_sq = np.sum((grid_points[:, valid_points] - x_si[:, i, None]) ** 2, axis=0)

            # Covered cost component (Temporal cost)
            covered_cost += np.sum(phi_q[valid_points] * (dist_sq / (Vr[i] ** 2))) * (res ** 2)

            # Centroid calculation
            c_v[i] = np.sum(grid_points[:, valid_points] * phi_q[valid_points] , axis=1)
            w_v[i] = np.sum(phi_q[valid_points] )

            if w_v[i] > 1e-6:
                c_v[i] /= w_v[i]
            else:
                c_v[i] = x_si[:, i]  # Fallback to current position

    # Total cost including both components
    total_cost = covered_cost
    if len(cost_history) > 1 and abs(total_cost - cost_history[-1]) < 0.001 and convergence == 0:
        convergence = k
    cost_history.append(total_cost)

    # Compute control inputs
    si_velocities = np.zeros((2, N))

    for i in range(N):
        if w_v[i] > 1e-6:
            # Calculate desired velocity with velocity constraints
            desired_velocity = k_gain * (c_v[i] - x_si[:, i])
            norm = np.linalg.norm(desired_velocity)
            if norm > 1e-6:
                scaling_factor = min(Vr[i] / norm, 1.0)
                si_velocities[:, i] = desired_velocity * scaling_factor
        else:
            # No valid points - stop the robot
            si_velocities[:, i] = np.zeros(2)

    # Transform to unicycle velocities
    dxu = si_to_uni_dyn(si_velocities, x)

    # Set velocities and step simulation
    r.set_velocities(np.arange(N), dxu)
    r.step()
        

# Save results and cleanup
final_positions = np.array([robot_trajectories[i][-1] for i in range(N)])  # Last recorded position
initial_positions = np.array([robot_trajectories[i][0] for i in range(N)])  # First recorded position
plot_robot_trajectories(robot_trajectories, initial_positions, final_positions, scenario_name="Case 10")
pd_cost = pd.DataFrame(cost_history)
pd_cost.to_csv("mobility_coverage_cost_10.csv", index=False)
with open("mobility_coverage_case_10.txt", "w") as f:
    f.write(str(convergence))

r.call_at_scripts_end()
