import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
from rps.utilities.graph import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


def standard_coverage(scenario_num):
    """
    Implements standard coverage control (Baseline 0) for the given scenario number.
    """
    # Scenario-specific parameters
    if scenario_num <= 6:
        N = 5  # Number of robots for scenarios 1-6
    else:
        N = 10  # Number of robots for scenarios 7-10

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

    # Density function (constant for this example)
    phi_q = np.ones(grid_points.shape[1])  # Uniform density

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
        plt.title(f"Robot Trajectories for Standard Coverage: Scenario {scenario_num}", fontsize=16)
        plt.xlabel("X Position", fontsize=12, fontweight='bold')
        plt.ylabel("Y Position", fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', bbox_to_anchor=(1.10, 1), borderaxespad=0.)
        plt.grid()

        # Save plot
        plt.savefig(f"standard_trajectories_case_{scenario_num}.png")
        plt.close()

    # Control gain
    k_gain = 1.0

    # Cost history and convergence tracking
    cost_history = []
    robot_trajectories = {i: [] for i in range(N)}  # Store (x, y) positions over time

    # Main simulation loop
    for k in range(iterations):
        x = r.get_poses()
        x_si = uni_to_si_states(x)[:2, :]  # (2, N)

        # Store position history
        for i in range(N):
            robot_trajectories[i].append(x_si[:, i].copy())

        # Compute standard Voronoi partition
        diff = grid_points[:, :, None] - x_si[:, None, :]  # (2, M, N)
        D_squared = np.sum(diff ** 2, axis=0)  # (M, N)
        robot_assignments = np.argmin(D_squared, axis=1)  # (M,)

        # Calculate covered cost and centroids
        c_v = np.zeros((N, 2))
        w_v = np.zeros(N)
        locational_cost = 0

        for i in range(N):
            # Points assigned to this robot
            valid_points = (robot_assignments == i)

            if np.any(valid_points):
                # Get distances for valid points
                dist_sq = D_squared[valid_points, i]

                # Locational cost component
                locational_cost += np.sum(phi_q[valid_points] * dist_sq) * (res ** 2)

                # Centroid calculation
                c_v[i] = np.sum(grid_points[:, valid_points] * phi_q[valid_points], axis=1)
                w_v[i] = np.sum(phi_q[valid_points])

                if w_v[i] > 0:
                    c_v[i] /= w_v[i]
                else:
                    c_v[i] = x_si[:, i]  # Fallback to current position

        # Check convergence
        if len(cost_history) > 1 and abs(locational_cost - cost_history[-1]) < 0.001 and convergence == 0:
            convergence = k

        cost_history.append(locational_cost)

        # Compute control inputs
        si_velocities = np.zeros((2, N))

        for i in range(N):
            # Calculate desired velocity
            si_velocities[:, i] = k_gain * (c_v[i] - x_si[:, i])

        # Transform to unicycle velocities
        dxu = si_to_uni_dyn(si_velocities, x)

        # Set velocities and step simulation
        r.set_velocities(np.arange(N), dxu)
        r.step()

    # Save results and cleanup
    final_positions = np.array([robot_trajectories[i][-1] for i in range(N)])  # Last recorded position
    initial_positions = np.array([robot_trajectories[i][0] for i in range(N)])  # First recorded position

    plot_robot_trajectories(robot_trajectories, initial_positions, final_positions)

    # Save cost history to CSV
    pd_cost = pd.DataFrame(cost_history)
    pd_cost.to_csv(f"standard_coverage_cost_{scenario_num}.csv", index=False)

    # Save convergence iteration to file
    with open(f"standard_coverage_case_{scenario_num}.txt", "w") as f:
        f.write(str(convergence))

    r.call_at_scripts_end()


# Run standard coverage for all 10 scenarios
for scenario in range(1, 11):
    print(f"Running standard coverage for scenario {scenario}")
    standard_coverage(scenario)
