import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import glob

# Create output directory for plots
output_dir = "coverage_plots"
os.makedirs(output_dir, exist_ok=True)

# Create subdirectories for different plot types
cost_dir = os.path.join(output_dir, "cost_plots")
trajectory_dir = os.path.join(output_dir, "trajectory_plots")
convergence_dir = os.path.join(output_dir, "convergence_plots")

os.makedirs(cost_dir, exist_ok=True)
os.makedirs(trajectory_dir, exist_ok=True)
os.makedirs(convergence_dir, exist_ok=True)

# Define scenarios
scenarios = [
    "No Heterogeneity",
    "Type-only",
    "Range-only",
    "Mobility-only",
    "Health-only",
    "All (N=5)",
    "Type-only (N=10)",
    "Type+Range (N=10)",
    "Type+Range+Mobility (N=10)",
    "All (N=10)"
]

# Define methods
methods = [
    "Standard",
    "Type",
    "Health",
    "Mobility",
    "Range",
    "Unified"
]

# Define cost metrics
cost_metrics = [
    "Hg",  # Locational cost
    "Hp",  # Power cost
    "Ht",  # Temporal cost
    "Hr",  # Range-Limited Cost
    "Sc"  # Generalized locational cost
]


# Function to find and load cost data files with flexible naming
def load_cost_data(method, scenario_num):
    """
    Attempts to load cost data with various possible filename patterns.
    Returns DataFrame if successful, None otherwise.
    """
    possible_patterns = [
        f"{method.lower()}_coverage_cost_{scenario_num}.csv",
        f"{method.lower()}_coverage_case_{scenario_num}.csv",
        f"{method.lower()}_coverage_results_{scenario_num}.csv",
        f"{method.lower()}_coverage_results_scenario{scenario_num}.csv",
        f"unified_coverage_results.csv" if method.lower() == "unified" else None
    ]

    # Remove None entries
    possible_patterns = [p for p in possible_patterns if p is not None]

    # Try each pattern
    for pattern in possible_patterns:
        if os.path.exists(pattern):
            print(f"Found file: {pattern}")
            try:
                df = pd.read_csv(pattern)
                # Check column structure
                if df.shape[1] == 0:
                    print(f"File {pattern} is empty")
                    continue

                # Print column names for debugging
                print(f"Columns in {pattern}: {df.columns.tolist()}")

                # Try to extract cost data based on column names
                if "Generalized_Cost" in df.columns:
                    return df["Generalized_Cost"]
                elif method.lower() + "_cost" in df.columns:
                    return df[method.lower() + "_cost"]
                elif df.shape[1] == 1:  # Single column without header
                    return df.iloc[:, 0]
                elif 0 in df.columns:  # Numbered column
                    return df[0]
                else:
                    print(f"Could not identify cost column in {pattern}")
            except Exception as e:
                print(f"Error reading {pattern}: {e}")

    # If we get here, no file was found or loaded successfully
    print(f"No valid cost data file found for {method} in scenario {scenario_num}")
    return None


# Function to plot cost metrics for a scenario
def plot_cost_metrics(scenario_num):
    scenario_name = scenarios[scenario_num - 1]
    print(f"\nProcessing cost plots for Scenario {scenario_num}: {scenario_name}")

    # For each cost metric
    for metric in cost_metrics:
        plt.figure(figsize=(10, 6))
        legend_entries = []

        # For each method
        for method in methods:
            print(f"  Loading {method} data for {metric} metric...")

            # Load the data
            cost_data = load_cost_data(method, scenario_num)

            if cost_data is not None:
                # Plot the cost over iterations
                plt.plot(range(len(cost_data)), cost_data, linewidth=2)
                legend_entries.append(method)
                print(f"  Successfully plotted {method} data")
            else:
                print(f"  Skipping {method} (no data)")

        if not legend_entries:
            print(f"  No data available for any method in {metric} plot")
            plt.close()
            continue

        plt.title(f"{metric} Cost Over Time - Scenario {scenario_num}: {scenario_name}", fontsize=14)
        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel(f"{metric} Cost", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(legend_entries)

        # Save the plot
        output_file = os.path.join(cost_dir, f"scenario_{scenario_num}_{metric}_cost.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"  Saved plot to {output_file}")
        plt.close()


# Function to find and copy trajectory plots
def find_trajectory_plots(scenario_num):
    """
    Finds and copies existing trajectory plots to the output directory.
    """
    print(f"\nProcessing trajectory plots for Scenario {scenario_num}")

    for method in methods:
        # Possible filename patterns
        possible_patterns = [
            f"{method.lower()}_trajectories_case_{scenario_num}.png",
            f"{method.lower()}_trajectory_case_{scenario_num}.png",
            f"{method.lower()}_trajectory_scenario_{scenario_num}.png"
        ]

        # Check if any of these files exist
        for pattern in possible_patterns:
            if os.path.exists(pattern):
                print(f"  Found trajectory plot: {pattern}")
                # Copy to output directory
                import shutil
                output_file = os.path.join(trajectory_dir, f"scenario_{scenario_num}_{method}_trajectory.png")
                shutil.copy(pattern, output_file)
                print(f"  Copied to {output_file}")
                break
        else:
            print(f"  No trajectory plot found for {method} in scenario {scenario_num}")


# Function to load convergence data
def load_convergence_data(scenario_num):
    """
    Loads convergence data for all methods in a scenario.
    """
    print(f"\nProcessing convergence data for Scenario {scenario_num}")
    convergence_data = {}

    for method in methods:
        # Possible filename patterns
        possible_patterns = [
            f"{method.lower()}_coverage_convergence_{scenario_num}.txt",
            f"{method.lower()}_coverage_case_{scenario_num}.txt",
            f"{method.lower()}_convergence_{scenario_num}.txt"
        ]

        # Try each pattern
        for pattern in possible_patterns:
            if os.path.exists(pattern):
                try:
                    with open(pattern, 'r') as f:
                        value = int(f.read().strip())
                        convergence_data[method] = value
                        print(f"  Loaded convergence data for {method}: {value} iterations")
                        break
                except Exception as e:
                    print(f"  Error reading {pattern}: {e}")
        else:
            print(f"  No convergence data found for {method}")

    return convergence_data


# Function to plot convergence iterations as bar chart
def plot_convergence(scenario_num):
    scenario_name = scenarios[scenario_num - 1]

    # Load convergence data
    convergence_data = load_convergence_data(scenario_num)

    if not convergence_data:
        print(f"  No convergence data available for Scenario {scenario_num}")
        return

    # Create bar plot
    plt.figure(figsize=(10, 6))

    methods_list = list(convergence_data.keys())
    iterations = list(convergence_data.values())

    # Use different colors for each method
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    bars = plt.bar(methods_list, iterations, color=colors[:len(methods_list)])

    # Add iteration numbers on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
                 f'{int(height)}', ha='center', va='bottom')

    plt.title(f"Convergence Iterations - Scenario {scenario_num}: {scenario_name}", fontsize=14)
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Iterations to Convergence", fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)

    # Save the plot
    output_file = os.path.join(convergence_dir, f"scenario_{scenario_num}_convergence.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved convergence plot to {output_file}")
    plt.close()


# Generate all plots
for scenario_num in range(1, 11):
    # Cost plots for each scenario
    plot_cost_metrics(scenario_num)

    # Find and copy trajectory plots
    find_trajectory_plots(scenario_num)

    # Convergence bar plot for each scenario
    plot_convergence(scenario_num)

print(f"\nAll plots have been generated and saved in the '{output_dir}' directory.")
