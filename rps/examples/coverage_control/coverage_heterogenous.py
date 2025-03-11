import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
from rps.utilities.graph import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import os


def run_unified_coverage(scenario_num):
    """
    Run the unified coverage controller for a specific scenario number.
    """
    # Create output directory for data
    os.makedirs("data", exist_ok=True)

    # Scenario-specific parameters
    if scenario_num <= 6:
        N = 5  # Number of robots for scenarios 1-6
    else:
        N = 10  # Number of robots for scenarios 7-10

    # Initialize parameters based on scenario
    sensor_types, healths, ranges, mobility = initialize_scenario_params(scenario_num, N)

    # Define initial conditions
    initial_positions = np.asarray([
        [1.25, 0.25, 0], [1, 0.5, 0], [1, -0.5, 0], [-1, -0.75, 0], [0.1, 0.2, 0],
        [0.2, -0.6, 0], [-0.75, -0.1, 0], [-1, 0, 0], [-0.8, -0.25, 0], [1.3, -0.4, 0]
    ])

    # Create Robotarium object
    r = robotarium.Robotarium(
        number_of_robots=N,
        show_figure=True,
        sim_in_real_time=True,
        initial_conditions=initial_positions[:N].T
    )

    # Simulation parameters
    iterations = 1000
    convergence = 0

    # Dynamics mapping
    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

    # Workspace parameters
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.0, 1.0
    res = 0.05

    # Generate grid points
    X, Y = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))
    grid_points = np.vstack([X.ravel(), Y.ravel()])  # (2, M)

    # Density function (constant for this example)
    phi_q = np.ones(grid_points.shape[1])  # Uniform density

    # Unified Voronoi partitioning
    def unified_voronoi_partition(x_si, v_i, Nj, Rrsi, hij):
        """
        Compute the unified Voronoi partition that considers all heterogeneities.
        """
        D_weighted = np.full((grid_points.shape[1], N), np.inf)

        for sensor_type, robots in Nj.items():
            if len(robots) > 0:
                sensing_range = Rrsi[sensor_type]
                for i, j in enumerate(robots):
                    dist = np.linalg.norm(grid_points - x_si[:, j, None], axis=0)
                    in_range = dist <= sensing_range[i]
                    S_i = np.full(dist.shape, np.inf)
                    S_i[in_range] = (dist[in_range] / v_i[j]) + (
                                dist[in_range] - hij[sensor_type][i])
                    D_weighted[:, j] = np.minimum(D_weighted[:, j], S_i)

        return np.argmin(D_weighted, axis=1), D_weighted

    # Calculate all cost metrics
    def calculate_costs(x_si, robot_assignments, D_weighted, v_i, Nj, Rrsi, hij):
        """
        Calculate all cost metrics as required by the project.
        """
        # Initialize costs
        locational_cost = 0.0  # Hg: Locational cost (BASELINE 1)
        power_cost = 0.0  # Hp: Power cost (BASELINE 2)
        temporal_cost = 0.0  # Ht: Temporal cost (BASELINE 3)
        range_limited_cost = 0.0  # Hr: Range-Limited Cost (BASELINE 4)
        generalized_cost = 0.0  # Sc: Generalized locational cost (proposed)

        # Calculate centroids and costs
        c_v = np.zeros((N, 2))
        w_v = np.zeros(N)

        for i in range(N):
            # Points assigned to this robot
            valid_points = (robot_assignments == i)

            if np.any(valid_points):
                # Get distances for valid points
                dist_sq = np.sum((grid_points[:, valid_points] - x_si[:, i, None]) ** 2, axis=0)
                dist = np.sqrt(dist_sq)

                # 1. Locational Cost (Hg) - BASELINE 1
                locational_cost += np.sum(phi_q[valid_points] * dist_sq) * (res ** 2)

                # 2. Power Cost (Hp) - BASELINE 2
                # Find sensor type for this robot
                for sensor_type, robots in Nj.items():
                    if i in robots:
                        idx = robots.index(i)
                        health = hij[sensor_type][idx]
                        power_cost += np.sum(phi_q[valid_points] * (dist_sq - health)) * (res ** 2)

                # 3. Temporal Cost (Ht) - BASELINE 3
                temporal_cost += np.sum(phi_q[valid_points] * (dist_sq / (v_i[i] ** 2))) * (res ** 2)

                # 4. Range-Limited Cost (Hr) - BASELINE 4
                for sensor_type, robots in Nj.items():
                    if i in robots:
                        idx = robots.index(i)
                        range_i = Rrsi[sensor_type][idx]

                        # Points within range
                        in_range = dist <= range_i
                        if np.any(in_range):
                            range_limited_cost += np.sum(phi_q[valid_points][in_range] *
                                                         dist_sq[in_range]) * (res ** 2)

                        # Points outside range but in Voronoi cell
                        out_of_range = ~in_range
                        if np.any(out_of_range):
                            range_limited_cost += np.sum(phi_q[valid_points][out_of_range] *
                                                         (range_i ** 2)) * (res ** 2)

                # In calculate_costs function
                # For generalized cost calculation
                # Make sure these arrays have compatible shapes
                points_to_use = valid_points & np.isfinite(D_weighted[:, i])
                if np.any(points_to_use):
                    generalized_cost += np.sum(phi_q[points_to_use] * D_weighted[points_to_use, i]) * (res ** 2)

                # Centroid calculation
                c_v[i] = np.sum(grid_points[:, valid_points] * phi_q[valid_points], axis=1)
                w_v[i] = np.sum(phi_q[valid_points])

                if w_v[i] > 0:
                    c_v[i] /= w_v[i]
                else:
                    c_v[i] = x_si[:, i]  # Default to current position if no points assigned

        return c_v, w_v, {
            'Hg': locational_cost,
            'Hp': power_cost,
            'Ht': temporal_cost,
            'Hr': range_limited_cost,
            'Sc': generalized_cost
        }

    def plot_robot_trajectories(robot_trajectories, initial_positions, final_positions):
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
        plt.title(f"Robot Trajectories for Unified Coverage: Scenario {scenario_num}", fontsize=16)
        plt.xlabel("X Position", fontsize=12, fontweight='bold')
        plt.ylabel("Y Position", fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', bbox_to_anchor=(1.10, 1), borderaxespad=0.)
        plt.grid()

        # Save plot
        plt.savefig(f"data/unified_trajectories_case_{scenario_num}.png")
        plt.close()

    # Control gain
    k_gain = 1.0

    # Cost history and convergence tracking
    cost_history = {
        'Hg': [],  # Locational cost
        'Hp': [],  # Power cost
        'Ht': [],  # Temporal cost
        'Hr': [],  # Range-Limited Cost
        'Sc': []  # Generalized locational cost
    }
    robot_trajectories = {i: [] for i in range(N)}  # Store (x, y) positions over time
    dist_robots = np.zeros(N)  # Distance traveled by each robot
    dist_sum = []  # Cumulative distance traveled by all robots

    # Main simulation loop
    for k in range(iterations):
        x = r.get_poses()
        x_si = uni_to_si_states(x)[:2, :]  # (2, N)

        # Store position history
        for i in range(N):
            robot_trajectories[i].append(x_si[:, i].copy())

        # Calculate distance traveled since last iteration
        if k > 0:
            for i in range(N):
                prev_pos = robot_trajectories[i][-2]
                curr_pos = robot_trajectories[i][-1]
                dist_robots[i] += np.linalg.norm(curr_pos - prev_pos)

        # Update cumulative distance
        dist_sum.append(np.sum(dist_robots))

        # Compute unified Voronoi partition
        robot_assignments, D_weighted = unified_voronoi_partition(
            x_si, mobility, sensor_types, ranges, healths
        )

        # Calculate costs and centroids
        centroids, weights, costs = calculate_costs(
            x_si, robot_assignments, D_weighted, mobility, sensor_types, ranges, healths
        )

        # Store costs
        for metric, value in costs.items():
            cost_history[metric].append(value)

        # Check convergence
        if k > 10:
            if abs(cost_history['Sc'][-1] - cost_history['Sc'][-10]) < 0.001 and convergence == 0:
                convergence = k
                print(f"Converged at iteration {k}")

        # Compute control inputs
        si_velocities = np.zeros((2, N))

        for i in range(N):
            # Calculate desired velocity with velocity constraints
            desired_velocity = k_gain * (centroids[i] - x_si[:, i])
            norm = np.linalg.norm(desired_velocity)

            if norm > 0:
                scaling_factor = min(mobility[i] / norm, 1.0)
                si_velocities[:, i] = desired_velocity * scaling_factor

        # Transform to unicycle velocities
        dxu = si_to_uni_dyn(si_velocities, x)

        # Set velocities and step simulation
        r.set_velocities(np.arange(N), dxu)
        r.step()

    # Save results and cleanup
    final_positions = np.array([robot_trajectories[i][-1] for i in range(N)])  # Last recorded position
    initial_positions = np.array([robot_trajectories[i][0] for i in range(N)])  # First recorded position

    # Plot trajectories
    plot_robot_trajectories(robot_trajectories, initial_positions, final_positions)

    # Save cost history to CSV
    df_results = pd.DataFrame({
        "Iteration": list(range(len(cost_history['Hg']))),
        "Hg": cost_history['Hg'],
        "Hp": cost_history['Hp'],
        "Ht": cost_history['Ht'],
        "Hr": cost_history['Hr'],
        "Sc": cost_history['Sc'],
        "dist_sum": dist_sum
    })
    df_results.to_csv(f"data/unified_coverage_cost_{scenario_num}.csv", index=False)

    # Save convergence iteration to file
    with open(f"data/unified_coverage_case_{scenario_num}.txt", "w") as f:
        f.write(str(convergence))

    r.call_at_scripts_end()


def initialize_scenario_params(scenario_num, N):
    """
    Initialize parameters for each scenario based on the project description.
    """
    # Default parameters
    sensor_types = {}
    healths = {}
    ranges = {}
    mobility = np.ones(N)

    if scenario_num == 1:  # No Heterogeneity
        sensor_types = {1: list(range(N))}
        healths = {1: [1.0] * N}
        ranges = {1: [1.0] * N}
        mobility = np.ones(N)

    elif scenario_num == 2:  # Type-only
        sensor_types = {1: [0, 1, 2], 2: [2, 3, 4]}
        healths = {1: [1.0, 1.0, 1.0], 2: [1.0, 1.0, 1.0]}
        ranges = {1: [1.0, 1.0, 1.0], 2: [1.0, 1.0, 1.0]}
        mobility = np.ones(N)

    elif scenario_num == 3:  # Range-only
        sensor_types = {1: list(range(N))}
        healths = {1: [1.0] * N}
        ranges = {1: [1.0, 1.0, 0.5, 1.0, 1.0]}
        mobility = np.ones(N)

    elif scenario_num == 4:  # Mobility-only
        sensor_types = {1: list(range(N))}
        healths = {1: [1.0] * N}
        ranges = {1: [1.0] * N}
        mobility = np.array([1.0, 1.0, 2.0, 1.0, 1.0])

    elif scenario_num == 5:  # Health-only
        sensor_types = {1: list(range(N))}
        healths = {1: [1.0, 1.0, 0.5, 1.0, 1.0]}
        ranges = {1: [1.0] * N}
        mobility = np.ones(N)

    elif scenario_num == 6:  # All (N=5)
        sensor_types = {1: [0, 1, 2], 2: [2, 3, 4]}
        healths = {1: [1.0, 0.5, 1.0], 2: [1.0, 0.5, 1.0]}
        ranges = {1: [1.0, 0.5, 1.0], 2: [1.0, 0.5, 1.0]}
        mobility = np.array([1.0, 1.0, 2.0, 1.0, 1.0])

    elif scenario_num == 7:  # Type-only (N=10)
        sensor_types = {1: [0, 1, 2, 3, 4], 2: [5, 6, 7, 8, 9]}
        healths = {1: [1.0] * 5, 2: [1.0] * 5}
        ranges = {1: [1.0] * 5, 2: [1.0] * 5}
        mobility = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    elif scenario_num == 8:  # Type+Range (N=10)
        sensor_types = {1: [0, 1, 2, 3, 4], 2: [5, 6, 7, 8, 9]}
        healths = {1: [1.0] * 5, 2: [1.0] * 5}
        ranges = {1: [1.0, 1.0, 0.5, 1.0, 1.0], 2: [1.0, 1.0, 0.5, 1.0, 1.0]}
        mobility = np.ones(N)

    elif scenario_num == 9:  # Type+Range+Mobility (N=10)
        sensor_types = {1: [0, 1, 2, 3, 4], 2: [5, 6, 7, 8, 9]}
        healths = {1: [1.0] * 5, 2: [1.0] * 5}
        ranges = {1: [1.0, 1.0, 0.5, 1.0, 1.0], 2: [1.0, 1.0, 0.5, 1.0, 1.0]}
        mobility = np.array([1,1,2,1,1,1,1,2,1,1])

    elif scenario_num == 10: # All
        sensor_types = {1: [0, 1, 2, 3, 4], 2: [5, 6, 7, 8, 9]}
        healths = {1: [1,1,0.5,1,1], 2: [1,1,0.5,1,1]}
        ranges = {1: [1,1,0.5,1,1], 2: [1,1,0.5,1,1]}
        mobility = np.array([1,1,2,1,1,1,1,2,1,1])

    return sensor_types, healths, ranges, mobility

for scenario_num in range(1, 11):
    run_unified_coverage(scenario_num)