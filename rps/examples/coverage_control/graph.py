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


def plot_cost_comparison(scenario_num):
    """
    Plot specific cost metrics between unified and each baseline:
    1. H_g in unified vs H_g in type
    2. H_p in unified vs H_p in health
    3. H_t in unified vs H_t in mobility
    4. H_r in unified vs H_r in range
    5. S_c in unified vs S_c in standard
    """
    scenario_name = scenarios[scenario_num - 1]
    print(f"\nProcessing cost comparisons for Scenario {scenario_num}: {scenario_name}")

    # Load unified data first
    unified_data = {}
    try:
        unified_filename = f"data/unified_coverage_cost_{scenario_num}.csv"
        if os.path.exists(unified_filename):
            unified_df = pd.read_csv(unified_filename)
            for metric in cost_metrics:
                if metric in unified_df.columns:
                    unified_data[metric] = unified_df[metric]
                    print(f"  Successfully loaded Unified {metric} data")
    except Exception as e:
        print(f"  Error loading Unified data: {e}")

    # Define specific comparisons
    comparisons = [
        ("Hg", "Type")
        ("Hp", "Health"),
        ("Ht", "Mobility"),
        ("Hr", "Range"),
        ("Sc", "Standard")
    ]

    # Create each comparison plot
    for metric, baseline in comparisons:
        if metric not in unified_data:
            print(f"  Missing {metric} data for Unified controller")
            continue

        # Load baseline data
        baseline_data = None
        baseline_filename = f"data/{baseline.lower()}_coverage_cost_{scenario_num}.csv"

        try:
            if os.path.exists(baseline_filename):
                baseline_df = pd.read_csv(baseline_filename)

                # Extract cost data from baseline
                if baseline_df.shape[1] == 1:  # Single column file
                    baseline_data = baseline_df.iloc[:, 0]
                elif 0 in baseline_df.columns:  # Numbered column
                    baseline_data = baseline_df[0]
                else:
                    print(f"  Could not identify cost column in {baseline_filename}")
                    continue
            else:
                print(f"  File {baseline_filename} does not exist")
                continue
        except Exception as e:
            print(f"  Error loading {baseline} data: {e}")
            continue

        # Create comparison plot
        if baseline_data is not None:
            plt.figure(figsize=(10, 6))

            # Truncate to the shorter length
            min_length = min(len(unified_data[metric]), len(baseline_data))

            # Plot both datasets
            plt.plot(range(min_length), unified_data[metric][:min_length],
                     linewidth=2, color='blue', label=f"Unified {metric}")
            plt.plot(range(min_length), baseline_data[:min_length],
                     linewidth=2, color='red', linestyle='--', label=f"{baseline} {metric}")

            plt.title(f"{metric} Cost Comparison - Unified vs {baseline} - Scenario {scenario_num}", fontsize=14)
            plt.xlabel("Iterations", fontsize=12)
            plt.ylabel(f"{metric} Cost", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Save the plot
            output_file = os.path.join(cost_dir, f"scenario_{scenario_num}_{metric}_unified_vs_{baseline.lower()}.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"  Saved plot to {output_file}")
            plt.close()
        else:
            print(f"  Missing data for comparison between Unified and {baseline}")


# Function to find and copy trajectory plots
def find_trajectory_plots(scenario_num):
    """
    Finds and copies existing trajectory plots to the output directory.
    """
    print(f"\nProcessing trajectory plots for Scenario {scenario_num}")

    for method in methods:
        # Possible filename patterns
        possible_patterns = [
            f"data/{method.lower()}_trajectories_case_{scenario_num}.png",
            f"{method.lower()}_trajectories_case_{scenario_num}.png",
            f"data/{method.lower()}_trajectory_case_{scenario_num}.png",
            f"{method.lower()}_trajectory_case_{scenario_num}.png"
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
            f"data/{method.lower()}_coverage_case_{scenario_num}.txt",
            f"{method.lower()}_coverage_case_{scenario_num}.txt",
            f"data/{method.lower()}_convergence_{scenario_num}.txt",
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
    plot_cost_comparison(scenario_num)

    # Find and copy trajectory plots
    find_trajectory_plots(scenario_num)

    # Convergence bar plot for each scenario
    plot_convergence(scenario_num)

print(f"\nAll plots have been generated and saved in the '{output_dir}' directory.")
