import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
from rps.utilities.barrier_certificates import create_single_integrator_barrier_certificate_with_boundary
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d  # Import Voronoi and plotting function
import matplotlib.pyplot as plt

# ============================
# Initialize Parameters
# ============================
N = 5  # Number of robots (update dynamically for scenarios)
iterations = 1000  # Max iterations
alpha = 0.5  # Weighting factor for velocity vs sensor health

# Environment bounds and resolution
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.0, 1.0
res = 0.05

# Scenario-specific parameters (update dynamically based on scenarios)
sensor_types = {1: [0, 1, 2], 2: [3, 4]}  # Sensor type assignments
healths = {1: {0: 1.0, 1: 1.0, 2: 1.0}, 2: {3: 1.0, 4: 1.0}}  # Health of sensors by type and robot index
ranges = {1: {0: 1.0, 1: 1.0, 2: 1.0}, 2: {3: 1.0, 4: 1.0}}  # Range of sensors by type and robot index
mobility = [1.0, 1.0, 2.0, 1.0, 1.0]  # Robot mobility (velocity limits)

# Initialize Robotarium with specified starting positions
initial_conditions = np.asarray([
    [1.25, 0.25, 0], [1, 0.5, 0], [1, -0.5, 0], [-1, -0.75, 0], [0.1, 0.2, 0]
])
r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=True,
    sim_in_real_time=True,
    initial_conditions=initial_conditions[:N].T
)

# Dynamics mapping and collision avoidance
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()


# ============================
# Helper Functions
# ============================

def calculate_voronoi_cost(x_si):
    """
    Calculate Voronoi partitions and coverage cost using the proposed cost function.
    Handles multiple sensor types with different characteristics.
    """
    c_v = np.zeros((N, 2))  # Centroids of Voronoi partitions
    w_v = np.zeros(N)  # Weights of Voronoi partitions
    total_generalized_cost = {"Sc": 0}

    for ix in np.arange(x_min, x_max + res, res):
        for iy in np.arange(y_min, y_max + res, res):
            distances = np.full(N, np.inf)  # Initialize distances with infinity

            for sensor_type, robots in sensor_types.items():
                for robot_idx in robots:
                    dist = np.sqrt((ix - x_si[0][robot_idx]) ** 2 + (iy - x_si[1][robot_idx]) ** 2)
                    if dist <= ranges[sensor_type][robot_idx]:  # Ensure point is within sensing range
                        distances[robot_idx] = alpha * dist / mobility[robot_idx] + \
                                               (1 - alpha) * (dist - healths[sensor_type][robot_idx])

            if np.all(distances == np.inf):  # Skip points outside all sensor ranges
                continue

            min_index = np.argmin(distances)  # Assign point to nearest robot based on Sc
            c_v[min_index][0] += ix  # Update centroid calculation
            c_v[min_index][1] += iy
            w_v[min_index] += res ** 2  # Update weight of Voronoi cell

            total_generalized_cost["Sc"] += distances[min_index] * res ** 2

    for i in range(N):
        if w_v[i] > 0:
            c_v[i] /= w_v[i]
        else:
            c_v[i] = x_si[:, i]

    return c_v, total_generalized_cost["Sc"]


def compute_control_inputs(x_si, centroids):
    """
    Compute velocity inputs for robots based on centroids of their Voronoi partitions.
    """
    si_velocities = np.zeros((2, N))
    k_gain = 1.0

    for i in range(N):
        desired_velocity = k_gain * (centroids[i] - x_si[:, i])
        norm_velocity = np.linalg.norm(desired_velocity)

        if norm_velocity > mobility[i]:  # Saturate velocity based on mobility limits
            desired_velocity *= mobility[i] / norm_velocity

        si_velocities[:, i] = desired_velocity

    return si_velocities


def plot_voronoi(x_si):
    """
    Plot the Voronoi diagram overlaid on the workspace.
    """
    points = x_si.T
    vor = Voronoi(points)

    fig, ax = plt.subplots(figsize=(8, 8))

    voronoi_plot_2d(vor, ax=ax)  # Use scipy's built-in plotting function

    ax.plot(points[:, 0], points[:, 1], 'ko', markersize=5)  # Robot positions

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title("Voronoi Diagram")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.grid(True)
    plt.show()


# ============================
# Main Simulation Loop
# ============================
cost_history = []
cumulative_distances = np.zeros(N)

for k in range(iterations):
    x_poses = r.get_poses()
    x_si = uni_to_si_states(x_poses)[:2]

    centroids, generalized_cost = calculate_voronoi_cost(x_si)

    si_velocities = compute_control_inputs(x_si, centroids)

    si_velocities = si_barrier_cert(si_velocities, x_si)  # Apply barrier certificate

    dxu = si_to_uni_dyn(si_velocities,x_poses)

    r.set_velocities(np.arange(N), dxu)

    cumulative_distances += np.linalg.norm(si_velocities * res ** k)

    cost_history.append(generalized_cost)

    if k % 100 == 99:  # Plot Voronoi regions every few iterations for visualization purposes
        plot_voronoi(x_si)

    if k > 10 and abs(cost_history[-1] - cost_history[-10]) < 1e-3:
        print(f"Converged at iteration {k}")


    r.step()

df_results = pd.DataFrame({
    "Iteration": list(range(len(cost_history))),
    "Generalized_Cost": cost_history,
})
df_results.to_csv("unified_coverage_results.csv", index=False)

r.call_at_scripts_end()
