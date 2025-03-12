import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    "Hr", 
]


def plot_cost_comparison(scenario_num):
    """
    Plot each cost metric for the unified controller against all baselines in the same graph.
    """
    scenario_name = scenarios[scenario_num - 1]
    print(f"\nProcessing cost comparisons for Scenario {scenario_num}: {scenario_name}")
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted', "solid"]
    colors = ['orange', 'green', 'red', 'purple', 'blue']

    # Load unified data
    unified_data = {}
    try:
        unified_filename = f"data/unified_coverage_cost_{scenario_num}.csv"
        if os.path.exists(unified_filename):
            unified_df = pd.read_csv(unified_filename)
            for metric in cost_metrics:
                if metric in unified_df.columns:
                    unified_data[metric] = unified_df[metric].astype(float)
                    print(f" Successfully loaded Unified {metric} data")
    except Exception as e:
        print(f" Error loading Unified data: {e}")

    # For each cost metric, create a comparison plot with all baselines
    for metric in cost_metrics:
        if metric not in unified_data:
            print(f" Missing {metric} data for Unified controller, skipping...")
            continue

        plt.figure(figsize=(12, 7))

        # Plot unified data
        plt.plot(range(len(unified_data[metric])), unified_data[metric],
                 linewidth=2, color='blue', label=f"Unified {metric}")

        # Plot each baseline on the same graph
        for i,method in enumerate(methods):
            if method == "Unified":
                continue

            # Load baseline data
            baseline_data = None
            baseline_filename = f"data/{method.lower()}_coverage_cost_{scenario_num}.csv"

            try:
                if os.path.exists(baseline_filename):
                    baseline_df = pd.read_csv(baseline_filename, usecols=['Hg', 'Hp', 'Ht', 'Hr'])
                    print(baseline_df)
                    if metric in baseline_df.columns:
                        baseline_data = baseline_df[metric].astype(float)
                        print(baseline_data.head())


                    if baseline_data is not None and len(baseline_data) > 0:
                        # Normalize data to start from the same point
                        min_length = min(len(unified_data[metric]), len(baseline_data))
                        if baseline_data[0] != 0:
                            normalized_data = (baseline_data[:min_length] / baseline_data[0]) * unified_data[metric][0]

                            plt.plot(range(min_length), normalized_data,
                                     linewidth=2, linestyle=linestyles[i], color=colors[i], label=f"{method} {metric}",alpha=0.7)
            except Exception as e:
                print(f" Error processing {method} data: {e}")

        plt.title(f"{metric} Cost Comparison - Scenario {scenario_num}: {scenario_name}", fontsize=16)
        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel(f"{metric} Cost", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save the plot
        output_file = os.path.join(cost_dir, f"scenario_{scenario_num}_{metric}_all_baselines.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f" Saved plot to {output_file}")
        plt.close()
        
    plt.figure(figsize=(12, 7))
    # Plot unified data
    # for col in ['Hg', 'Hp', 'Ht', 'Hr', 'Sc']:
    #     normalized_data = (baseline_data[:min_length] / baseline_data[0]) * unified_data[metric][0]

    #     plt.plot(range(min_length), normalized_data,linewidth=2, linestyle=linestyles[i], color=colors[i], label=f"{method} {metric}",alpha=0.7)
    #     plt.plot(unified_df['Iteration'], unified_df[col], label=col)
    normalized_df = unified_df.copy()
    for col in ['Hg', 'Hp', 'Ht', 'Hr', 'Sc']:
        # normalized_data = (baseline_data[:min_length] / baseline_data[0]) * unified_data[metric][0]
        min_val = unified_df[col].min()
        max_val = unified_df[col].max()
        normalized_df[col] = (unified_df[col] - min_val) / (max_val - min_val)

    # Plot the normalized data
    plt.figure(figsize=(12, 7))

    for i, col in enumerate(['Hg', 'Hp', 'Ht', 'Hr', 'Sc']):
        name = ""
        if col == "Hg":
            name = "Type"
        elif col == "Hp":
            name = "Health"
        elif col == "Ht":
            name = "Mobility"
        elif col == "Hr":
            name = "Range"
        elif col == "Sc":
            name = "Unified"
        #plt.plot(unified_df['Iteration'], normalized_df[col], label=f"{name} Sc")
        plt.plot(unified_df['Iteration'], normalized_df[col], linewidth=2, linestyle=linestyles[i], color=colors[i], label=f"{name} Sc",alpha=0.7)
        print("HERE")
    plt.title(f"Sc Cost Comparison - Scenario {scenario_num}: {scenario_name}", fontsize=16)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel(f"Sc Cost", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel("Iteration")
    output_file = os.path.join(cost_dir, f"scenario_{scenario_num}_Sc_all_baselines.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f" Saved plot to {output_file}")
    plt.close()
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
                print(f" Found trajectory plot: {pattern}")
                # Copy to output directory
                import shutil
                output_file = os.path.join(trajectory_dir, f"scenario_{scenario_num}_{method}_trajectory.png")
                shutil.copy(pattern, output_file)
                print(f" Copied to {output_file}")
                break
        else:
            print(f" No trajectory plot found for {method} in scenario {scenario_num}")

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
                        print(f" Loaded convergence data for {method}: {value} iterations")
                        break
                except Exception as e:
                    print(f" Error reading {pattern}: {e}")
        else:
            print(f" No convergence data found for {method}")

    return convergence_data

# Function to plot convergence iterations as bar chart
def plot_convergence(scenario_num):
    scenario_name = scenarios[scenario_num - 1]

    # Load convergence data
    convergence_data = load_convergence_data(scenario_num)

    if not convergence_data:
        print(f" No convergence data available for Scenario {scenario_num}")
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
    print(f" Saved convergence plot to {output_file}")
    plt.close()

# Generate all plots
for scenario_num in range(1, 11):
    # Cost plots for each scenario
    plot_cost_comparison(scenario_num)

    # Find and copy trajectory plots
    #find_trajectory_plots(scenario_num)

    # Convergence bar plot for each scenario
    #plot_convergence(scenario_num)

print(f"\nAll plots have been generated and saved in the '{output_dir}' directory.")
