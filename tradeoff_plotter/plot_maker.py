# %%
import matplotlib.pyplot as plt
import numpy as np


def create_plot(plot_name, data_dict, assumption_dict):
    fig, ax = plt.subplots(dpi=300)
    ax.set_title(plot_name)  # Add figure title

    # Add the linear function in the background.
    x_background = np.linspace(0, 12, 400)
    y_background = -1 * x_background + 12.55
    ax.plot(
        x_background, y_background, "lightgray", alpha=0.2, linewidth=5
    )  # Red, transparent line
    for label, (position, color) in assumption_dict.items():
        x_index = len(x_background) - len(x_background) * position
        x_value = x_background[int(x_index)]
        y_value = -1 * x_value + 12.55

        ax.plot(x_value, y_value, color=color, marker="o", alpha=1, markersize=20)

        # Get the width of the text
        ax.text(x_value, y_value - (1), label, ha="center", fontsize=8, alpha=0.2)

        # text.text(x_value, y_value - 0.5, label, ha="center", fontsize=8, alpha=0.2)

    # Plot the black circles with labels below
    for label, (x, y) in data_dict.items():
        ax.plot(x, 14 - y, "ko")  # Black circle
        text = ax.text(x, 14 - y - 0.75, label, rotation=0, ha="center")
        # height = text.get_window_extent().height
        # print(height)
        # Move text down the y axis by width
        # text.set_position((text.get_position()[0], text.get_position()[1] - height/100))

    ax.set_xlim([-1, 13])  # Set x-axis limits
    ax.set_ylim([-1, 14])  # Set y-axis limits
    ax.set_xlabel("Interpretability Score")  # X-Axis label
    ax.set_ylabel("Performance Score")  # Y-Axis label

    # Show only the highest and lowest x-tick
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    # Show only the highest and lowest y-tick
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])

    plt.show()
    fig.savefig(f"tradeoff_plotter/tradeoff_{plot_name}.svg")


# %%
# Performance Rank comes from HPO Performance table
average_rank_data = {
    "P-Splines": [7, 7.63],
    "TP-Splines": [8, 7.55],
    "EBM": [6, 4.05],
    "NAM": [3, 10.08],
    "GAMI-Net": [11, 6.83],
    "ExNN": [3, 11.75],
    "IGANN": [9, 6.30],
    "LR": [12, 11.08],
    "DT*": [3, 11.55],
    "RF": [1, 4.83],
    "XGB": [2, 4.30],
    "CatBoost": [2, 2.85],
    "MLP": [0, 6.48],
    "TabNet": [2, 9.75],
}

regression_rank_data = {
    "P-Splines": [7, 8.10],
    "TP-Splines": [8, 8.80],
    "EBM": [6, 4.75],
    "NAM": [3, 11.20],
    "GAMI-Net": [11, 6.90],
    "ExNN": [3, 11.65],
    "IGANN": [9, 6.65],
    "LR": [12, 10.80],
    "DT*": [3, 10.50],
    "RF": [1, 4.30],
    "XGB": [2, 3.80],
    "CatBoost": [2, 2.30],
    "MLP": [0, 6.00],
    "TabNet": [2, 9.25],
}

classification_rank_data = {
    "P-Splines": [7, 7.15],
    "TP-Splines": [8, 6.30],
    "EBM": [6, 3.35],
    "NAM": [3, 8.95],
    "GAMI-Net": [11, 6.75],
    "ExNN": [3, 11.85],
    "IGANN": [9, 5.95],
    "LR": [12, 11.35],
    "DT*": [3, 12.60],
    "RF": [1, 5.35],
    "XGB": [2, 4.80],
    "CatBoost": [2, 3.40],
    "MLP": [0, 6.95],
    "TabNet": [2, 10.25],
}

usecase_monotonicity_rank_data = {
    "P-Splines": [7, 7.63],
    "TP-Splines": [2, 7.55],
    "EBM": [6, 4.05],
    "NAM": [1, 10.08],
    "GAMI-Net": [7, 6.83],
    "ExNN": [0, 11.75],
    "IGANN": [3, 6.30],
    "LR": [12, 11.08],
    "DT*": [1, 11.55],
    "RF": [0, 4.83],
    "XGB": [5, 4.30],
    "CatBoost": [5, 2.85],
    "MLP": [0, 6.48],
    "TabNet": [0, 9.75],
}

usecase_interpretability_rank_data = {
    "P-Splines": [10, 7.63],
    "TP-Splines": [10, 7.55],
    "EBM": [8, 4.05],
    "NAM": [4, 10.08],
    "GAMI-Net": [12, 6.83],
    "ExNN": [0, 11.75],
    "IGANN": [12, 6.30],
    "LR": [12, 11.08],
    "DT*": [4, 11.55],
    "RF": [0, 4.83],
    "XGB": [0, 4.30],
    "CatBoost": [0, 2.85],
    "MLP": [0, 6.48],
    "TabNet": [0, 9.75],
}

assumption_data = {
    "Neural Networks": [0.95, "#ced4da"],
    "Tree Ensemble": [0.75, "#dee2e6"],
    "Decision Trees": [0.25, "#e9ecef"],
    "Linear Regression": [0.05, "#f8f9fa"],
}

# create_plot("Average", average_rank_data, assumption_data)
# create_plot("Regression", regression_rank_data, assumption_data)
# create_plot("Classification", classification_rank_data, assumption_data)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), dpi=300)

create_plot(
    "Results for Classification and Regression Tasks",
    average_rank_data,
    assumption_data,
)
create_plot("Results for Regression Tasks", regression_rank_data, assumption_data)
create_plot(
    "Results for Classification Tasks", classification_rank_data, assumption_data
)

create_plot(
    "Results for Weighted Scenario 'Monotonicity'",
    usecase_monotonicity_rank_data,
    assumption_data,
)

create_plot(
    "Results for Weighted Scenario 'Simplicity'",
    usecase_interpretability_rank_data,
    assumption_data,
)

plt.tight_layout()
plt.show()


# %%
