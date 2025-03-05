import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
from rps.utilities.graph import *
import numpy as np
import pandas as pd

# Number of robots
N = 5
 
L = completeGL(N)

# Define initial conditions
initial_conditions = np.array([
    [1.25, 1.0, 1.0, -1.0, 0.1],  # x positions
    [0.25, 0.5, -0.5, -0.75, 0.2],  # y positions
    [0.0, 0.0, 0.0, 0.0, 0.0]  # orientations
])

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

# Generate grid points
X, Y = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))
grid_points = np.vstack([X.ravel(), Y.ravel()])  # (2, M)

# Robot parameters (heterogeneous)
R_sr = np.array([1.0, 1.0, 0.5, 1.0, 1.0])  # Sensing ranges
Vr = np.array([1.0, 1.2, 0.8, 1.5, 1.0])  # Max velocities (heterogeneous speeds)
wi = [1, 1, 1, 1, 1]  # Weights for each robot
# Density function (constant for this example)
phi_q = np.ones(grid_points.shape[1])  # Uniform density


# Multiplicatively weighted Voronoi partitioning
def weighted_voronoi_partition(x_si, v_i):
    """Compute the multiplicatively weighted Voronoi partition."""
    D_weighted = np.zeros((grid_points.shape[1], N))
    for i in range(N):
        dist = np.linalg.norm(grid_points - x_si[:, i, None], axis=0) ** 2
        D_weighted[:, i] = dist - wi[i]

    return np.argmin(D_weighted, axis=1)


# Position Gain
kp = 1.0
# Weight Gain
kw = 1.0

# Cost history and convergence tracking
cost_history = []
prev_smoothed_cost = None

# Main simulation loop
for k in range(iterations):
    x = r.get_poses()
    x_si = uni_to_si_states(x)[:2, :]  # (2, N)

    # Compute multiplicatively weighted Voronoi partition
    robot_assignments = weighted_voronoi_partition(x_si, wi)

    # Initialize cost components
    covered_cost = 0.0

    # Calculate covered cost and centroids
    c_v = np.zeros((N, 2))
    w_v = np.zeros(N)

    for i in range(N):
        # Points assigned to this robot
        valid_points = (robot_assignments == i)

        if np.any(valid_points):
            # Get distances and weights for valid points
            dist_sq = np.sum((grid_points[:, valid_points] - x_si[:, i, None]) ** 2, axis=0)

            # Power Cost
            covered_cost += 0.5 * np.sum((dist_sq - wi[i]) * phi_q[valid_points]) * (res ** 2)

            # Centroid calculation
            c_v[i] = np.sum(grid_points[:, valid_points] * phi_q[valid_points] , axis=1)
            w_v[i] = np.sum(phi_q[valid_points])

            if w_v[i] > 1e-6:
                c_v[i] /= w_v[i]
            else:
                c_v[i] = x_si[:, i]  # Fallback to current position

    # Total cost including both components
    total_cost = covered_cost
    cost_history.append(total_cost)

    # Compute control inputs
    si_velocities = np.zeros((2, N))
    w_dot = np.zeros(N)

    for i in range(N):
        if w_v[i] > 1e-6:
            # Compute position update
            si_velocities[:, i] = kp * (c_v[i] - x_si[:, i])

            # Compute weight update
            weight_update = 0
            for j in range(N):
                if j != i:
                    weight_update += (wi[i] - wi[j] )  # Health-based adjustment
            w_dot[i] = (kw / (2 * w_v[i])) * weight_update

    # Update weights
    wi += w_dot * 0.1  # Small step for weight adaptation
    wi = np.clip(wi, 0.1, 2.0)  # Prevent negative weights

    # Transform to unicycle velocities
    dxu = si_to_uni_dyn(si_velocities, x)

    # Set velocities and step simulation
    try:
        r.set_velocities(np.arange(N), dxu)
        r.step()

        # Enhanced convergence check
        if k >= MIN_ITERATIONS:
            smoothed_cost = np.mean(cost_history[-10:])
            if prev_smoothed_cost is not None:
                rel_change = abs(smoothed_cost - prev_smoothed_cost) / max(abs(prev_smoothed_cost), 1e-6)

                if rel_change < CONVERGENCE_THRESHOLD:
                    print(f"Converged at iteration {k} with relative change {rel_change:.6f}")
                    break
            prev_smoothed_cost = smoothed_cost

        # Detailed debugging output
        print(f"Iter {k:03d}: Cost = {total_cost:.3f} | "
              f"Δ = {rel_change:.2e}" if k > MIN_ITERATIONS else f"Iter {k:03d}: Cost = {total_cost:.3f}")
        print(f"Robot Positions:\n{np.round(x_si, 3)}")
        print(f"Velocities:\n{np.round(si_velocities, 3)}\n")

    except Exception as e:
        print(f"Error at iter {k}: {str(e)}")
        break

# Save results and cleanup
pd_cost = pd.DataFrame(cost_history)
pd_cost.to_csv("quality.csv", index=False)
r.call_at_scripts_end()
