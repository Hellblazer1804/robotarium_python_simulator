import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
from rps.utilities.graph import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import os


def health_coverage(scenario_num):
    """
    Implements health-based coverage control for the given scenario number.
    """
    # Create output directory for data
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)


    # Scenario-specific parameters
    if scenario_num <= 6:
        N = 5  # Number of robots for scenarios 1-6
    else:
        N = 10  # Number of robots for scenarios 7-10

    L = completeGL(N)

    # Initialize parameters based on scenario
    sensor_types, healths, ranges, mobility = initialize_scenario_params(scenario_num, N)
    health_weights = [
        healths[1][i] if 1 in healths and i in healths[1] else healths[2][i] if 2 in healths and i in healths[
            2] else 1.0 for i in range(N)]

    # Define initial conditions based on the provided positions
    initial_positions = np.asarray([
        [1.25, 0.25, 0], [1, 0.5, 0], [1, -0.5, 0], [-1, -0.75, 0], [0.1, 0.2, 0],
        [0.2, -0.6, 0], [-0.75, -0.1, 0], [-1, 0, 0], [-0.8, -0.25, 0], [1.3, -0.4, 0]
    ])

    # Create Robotarium object
    r = robotarium.Robotarium(
        number_of_robots=N,
        show_figure=True,
        sim_in_real_time=False,
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

    # Multiplicatively weighted Voronoi partitioning
    def weighted_voronoi_partition(x_si, wi):
        """Compute the multiplicatively weighted Voronoi partition."""
        D_weighted = np.zeros((grid_points.shape[1], N))
        for i in range(N):
            dist = np.linalg.norm(grid_points - x_si[:, i, None], axis=0) ** 2
            D_weighted[:, i] = dist - wi[i]
        return np.argmin(D_weighted, axis=1)

    def calculate_all_costs(x_si, robot_assignments, D_squared, health_weights,phi_q):
        """
        Calculate all cost metrics for comprehensive comparison.
        """
        # Initialize costs
        locational_cost = 0.0  # Hg: Locational cost (BASELINE 1)
        power_cost = 0.0  # Hp: Power cost (BASELINE 2)
        temporal_cost = 0.0  # Ht: Temporal cost (BASELINE 3)
        range_limited_cost = 0.0  # Hr: Range-Limited Cost (BASELINE 4)

        # Calculate centroids and costs
        c_v = np.zeros((N, 2))
        w_v = np.zeros(N)

        # Calculate linear distances
        D_linear = np.sqrt(D_squared)

        # Find which points are within each robot's sensing range
        within_range = np.zeros((D_squared.shape[0], N), dtype=bool)
        for sensor_type, robots in sensor_types.items():
            for i, robot_idx in enumerate(robots):
                if robot_idx < N:
                    within_range[:, robot_idx] = D_linear[:, robot_idx] <= ranges[sensor_type][i]

        for i in range(N):
            # Points assigned to this robot
            valid_points = (robot_assignments == i)

            if np.any(valid_points):
                # Get distances for valid points
                dist_sq = D_squared[valid_points, i]

                # 1. Locational Cost (Hg) - BASELINE 0
                locational_cost += np.sum(phi_q[valid_points] * dist_sq) * (res ** 2)

                # 2. Power Cost (Hp) - BASELINE 2
                power_cost += np.sum(phi_q[valid_points] * (dist_sq - health_weights[i])) * (res ** 2)

                # 3. Temporal Cost (Ht) - BASELINE 3
                temporal_cost += np.sum(phi_q[valid_points] * (dist_sq / (mobility[i] ** 2))) * (res ** 2)

                # 4. Range-Limited Cost (Hr) - BASELINE 4
                in_range = within_range[valid_points, i]
                if np.any(in_range):
                    range_limited_cost += np.sum(phi_q[valid_points][in_range] *
                                                 dist_sq[in_range]) * (res ** 2)
                out_of_range = ~in_range
                if np.any(out_of_range):
                    for sensor_type, robots in sensor_types.items():
                        if i in robots:
                            idx = robots.index(i)
                            range_i = ranges[sensor_type][idx]
                            range_limited_cost += np.sum(phi_q[valid_points][out_of_range] *
                                                         (range_i ** 2)) * (res ** 2)
                # Centroid calculation
                c_v[i] = np.sum(grid_points[:, valid_points] * phi_q[valid_points], axis=1)
                w_v[i] = np.sum(phi_q[valid_points])

                if w_v[i] > 1e-6:
                    c_v[i] /= w_v[i]
                else:
                    c_v[i] = x_si[:, i]  # Fallback to current position

        return c_v, w_v, {
            'Hg': locational_cost,
            'Hp': power_cost,
            'Ht': temporal_cost,
            'Hr': range_limited_cost
        }

    def plot_robot_trajectories(robot_trajectories, initial_positions, final_positions, sensor_types, scenario_num):
        """
        Plots the robot trajectories along with their final Voronoi partitions,
        with border colors based on sensor type.
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Compute Voronoi partition for final positions
        vor = Voronoi(final_positions)

        # Define colors for sensor types
        sensor_type_colors = {
            1: 'red',  # Sensor Type 1
            2: 'blue',  # Sensor Type 2
            'default': 'black'  # Default color
        }

        # Iterate through Voronoi ridges (edges) to draw them with colors based on robot sensor type
        for ridge_index, ridge in enumerate(vor.ridge_vertices):
            # Check if this ridge is infinite
            if -1 in ridge:
                continue  # Skip infinite ridges

            # Get vertices of the ridge
            v1, v2 = vor.vertices[ridge]

            # Determine robots associated with this ridge
            p1, p2 = vor.ridge_points[ridge_index]

            # Determine sensor type for robot p1 (primary robot for this region)
            robot_sensor_type = None
            for sensor_type, robots in sensor_types.items():
                if p1 in robots:
                    robot_sensor_type = sensor_type
                    break

            # Choose color based on sensor type
            color = sensor_type_colors.get(robot_sensor_type, sensor_type_colors['default'])

            # Plot the ridge as a solid line (since it is a finite edge)
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], color=color, linestyle='-', linewidth=1.5)

        # Plot robot trajectories
        for i in range(len(robot_trajectories)):
            trajectory = np.array(robot_trajectories[i])

            # Mark initial and final positions
            ax.scatter(initial_positions[i, 0], initial_positions[i, 1], c='black', marker='o', s=50,
                       label="Start" if i == 0 else "")
            ax.scatter(final_positions[i, 0], final_positions[i, 1], c='red', marker='x', s=50,
                       label="End" if i == 0 else "")

            # Plot trajectory
            ax.plot(trajectory[:, 0], trajectory[:, 1], label=f"Robot {i}", linewidth=2)

        # Add a legend for sensor types
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Sensor Type 1'),
            Line2D([0], [0], color='blue', lw=2, label='Sensor Type 2'),
            Line2D([0], [0], color='black', lw=2, label='Default/Unknown Sensor Type')
        ]

        # Set axis labels and titles
        ax.set_xlabel("X Position", fontsize=12, fontweight='bold')
        ax.set_ylabel("Y Position", fontsize=12, fontweight='bold')
        ax.set_title(f"Robot Trajectories for Health Coverage: Scenario {scenario_num}", fontsize=14)
        ax.grid()

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        ax.legend(loc='upper left', title="Robots")
        # Configure plot
        plt.tight_layout()  # Adjust layout to fit everything in

        # Save plot
        plt.savefig(f"data/health_trajectories_case_{scenario_num}.png")
        plt.close()

    # Control gain
    k_gain = 1.0

    # Cost history and convergence tracking
    cost_history = {
        'Hg': [],  # Locational cost
        'Hp': [],  # Power cost
        'Ht': [],  # Temporal cost
        'Hr': []  # Range-Limited Cost
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

        # Compute multiplicatively weighted Voronoi partition
        robot_assignments = weighted_voronoi_partition(x_si, health_weights)

        # Compute distances between all robots and grid points
        diff = grid_points[:, :, None] - x_si[:, None, :]  # (2, M, N)
        D_squared = np.sum(diff ** 2, axis=0)  # (M, N)

        # Calculate costs and centroids
        phiq = np.ones(D_squared.shape[0])
        centroids, weights, costs = calculate_all_costs(x_si, robot_assignments, D_squared, health_weights,phiq)

        # Store costs
        for metric, value in costs.items():
            cost_history[metric].append(value)

        # Check convergence
        if k > 10 and abs(cost_history['Hg'][-1] - cost_history['Hg'][-10]) < 0.001 and convergence == 0:
            convergence = k
            print(f"Converged at iteration {k}")

        # Compute control inputs
        si_velocities = np.zeros((2, N))

        for i in range(N):
            # Calculate desired velocity
            si_velocities[:, i] = k_gain * (centroids[i] - x_si[:, i])

        # Transform to unicycle velocities
        dxu = si_to_uni_dyn(si_velocities, x)

        # Set velocities and step simulation
        r.set_velocities(np.arange(N), dxu)
        r.step()

    # Save results and cleanup
    final_positions = np.array([robot_trajectories[i][-1] for i in range(N)])  # Last recorded position
    initial_positions = np.array([robot_trajectories[i][0] for i in range(N)])  # First recorded position

    plot_robot_trajectories(robot_trajectories, initial_positions, final_positions, sensor_types,
                            f"Scenario {scenario_num}")

    pd_cost = pd.DataFrame(cost_history)
    pd_cost.to_csv(f"data/health_coverage_cost_{scenario_num}.csv", index=False)

    with open(f"data/health_coverage_case_{scenario_num}.txt", "w") as f:
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
        if N == 5:
            sensor_types = {1: [0, 1, 2], 2: [2, 3, 4]}
            healths = {1: [1.0, 1.0, 1.0], 2: [1.0, 1.0, 1.0]}
            ranges = {1: [1.0, 1.0, 1.0], 2: [1.0, 1.0, 1.0]}
        else:
            sensor_types = {1: [0, 1, 2, 3, 4], 2: [5, 6, 7, 8, 9]}
            healths = {1: [1.0] * 5, 2: [1.0] * 5}
            ranges = {1: [1.0] * 5, 2: [1.0] * 5}
        mobility = np.ones(N)
    elif scenario_num == 3:  # Range-only
        sensor_types = {1: list(range(N))}
        healths = {1: [1.0] * N}
        if N == 5:
            ranges = {1: [1.0, 1.0, 0.5, 1.0, 1.0]}
        else:
            ranges = {1: [1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]}
        mobility = np.ones(N)
    elif scenario_num == 4:  # Mobility-only
        sensor_types = {1: list(range(N))}
        healths = {1: [1.0] * N}
        ranges = {1: [1.0] * N}
        if N == 5:
            mobility = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
        else:
            mobility = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0])
    elif scenario_num == 5:  # Health-only
        sensor_types = {1: list(range(N))}
        if N == 5:
            healths = {1: [1.0, 1.0, 0.5, 1.0, 1.0]}
        else:
            healths = {1: [1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]}
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
        mobility = np.array([1.0] * N)
    elif scenario_num == 8:  # Type+Range (N=10)
        sensor_types = {1: [0, 1, 2, 3, 4], 2: [5, 6, 7, 8, 9]}
        healths = {1: [1.0] * 5, 2: [1.0] * 5}
        ranges = {1: [1.0, 1.0, 0.5, 1.0, 1.0], 2: [1.0, 1.0, 0.5, 1.0, 1.0]}
        mobility = np.ones(N)
    elif scenario_num == 9:  # Type+Range+Mobility (N=10)
        sensor_types = {1: [0, 1, 2, 3, 4], 2: [5, 6, 7, 8, 9]}
        healths = {1: [1.0] * 5, 2: [1.0] * 5}
        ranges = {1: [1.0, 1.0, 0.5, 1.0, 1.0], 2: [1.0, 1.0, 0.5, 1.0, 1.0]}
        mobility = np.array([1, 1, 2, 1, 1, 1, 1, 2, 1, 1])
    elif scenario_num == 10:  # All
        sensor_types = {1: [0, 1, 2, 3, 4], 2: [5, 6, 7, 8, 9]}
        healths = {1: [1, 1, 0.5, 1, 1], 2: [1, 1, 0.5, 1, 1]}
        ranges = {1: [1, 1, 0.5, 1, 1], 2: [1, 1, 0.5, 1, 1]}
        mobility = np.array([1, 1, 2, 1, 1, 1, 1, 2, 1, 1])

    return sensor_types, healths, ranges, mobility


# Run health coverage for all 10 scenarios
for scenario in range(1, 11):
    print(f"Running health coverage for scenario {scenario}")
    health_coverage(scenario)
