import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
import matplotlib as mpl

# Use LaTeX for text rendering
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
def plot_analysis_multiple_lines(df, fixed_vars, x_var, varying_var):
    """
    Plots multiple lines of P_AD against a specified variable on the x-axis, 
    with each line representing a unique value of the varying variable.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        fixed_vars (dict): Dictionary of variables to fix with their values, e.g., {'delta': 0.786, 'lambda': 0.132}.
        x_var (str): The variable to plot on the x-axis.
        varying_var (str): The variable to vary, creating multiple lines.
    """
    # Validate inputs
    if 'P_AD' not in df.columns:
        raise ValueError("The dataframe must contain the column 'P_AD' for plotting.")
    if x_var not in df.columns:
        raise ValueError(f"The specified x-axis variable '{x_var}' is not in the dataframe.")
    if varying_var not in df.columns:
        raise ValueError(f"The specified varying variable '{varying_var}' is not in the dataframe.")
    for var in fixed_vars:
        if var not in df.columns:
            raise ValueError(f"The fixed variable '{var}' is not in the dataframe.")
    
    # Filter the dataframe based on fixed variables
    filtered_df = df.copy()
    for var, value in fixed_vars.items():
        filtered_df = filtered_df[filtered_df[var] == value]
    
    # Check if there is any data left after filtering
    if filtered_df.empty:
        print(f"No data available for the specified fixed variables: {fixed_vars}")
        return
    
    # Get unique values of the varying variable
    unique_values = sorted(filtered_df[varying_var].unique())    
    # Normalize unique values for colormap
    norm = plt.Normalize(min(unique_values), max(unique_values))
    cmap = cm.get_cmap('viridis')
    
    # Plot each unique value as a separate line
    plt.figure(figsize=(15, 10))
    for value in unique_values:
        subset = filtered_df[filtered_df[varying_var] == value]
        if not subset.empty:
            plt.plot(
                subset[x_var], 
                subset['P_GT'], 
                marker='o', 
                label=f"{varying_var}={value:.3f}", 
                color=cmap(norm(value)),
            )
    
    # Title and labels
    title = "Fixed Parameters: " + ", ".join(f"{key}={value}" for key, value in fixed_vars.items())
    plt.title(f"Analysis of {x_var} vs P_GT with Varying {varying_var}\n{title}")
    plt.xlabel(x_var)
    plt.ylabel('P_GT')
    plt.legend(title=varying_var)
    plt.grid()
    plt.show()

def plot_analysis_average_and_vary(df, y_var, averaging_vars, x_var, varying_var):
    """
    Plots multiple lines of P_AD averaged over specified variables against a specified x-axis variable, 
    with each line representing a unique value of the varying variable.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        averaging_vars (list): List of variables to average over, e.g., ['delta', 'lambda'].
        x_var (str): The variable to plot on the x-axis.
        varying_var (str): The variable to vary, creating multiple lines.
    """
    # Validate inputs
    if y_var not in df.columns:
        raise ValueError(f"The dataframe must contain the column {y_var} for plotting.")
    if x_var not in df.columns:
        raise ValueError(f"The specified x-axis variable '{x_var}' is not in the dataframe.")
    if varying_var not in df.columns:
        raise ValueError(f"The specified varying variable '{varying_var}' is not in the dataframe.")
    for var in averaging_vars:
        if var not in df.columns:
            raise ValueError(f"The averaging variable '{var}' is not in the dataframe.")
    
    # Group by the x_var and varying_var, averaging over the averaging_vars
    grouped = df.groupby([x_var, varying_var]).mean(numeric_only=True).reset_index()
    
    # Get unique values of the varying variable
    unique_values = sorted(grouped[varying_var].unique())
    norm = plt.Normalize(min(unique_values), max(unique_values))
    cmap = cm.get_cmap('viridis')
    # Plot each unique value as a separate line
    plt.figure(figsize=(15, 10))
    for value in unique_values:
        subset = grouped[grouped[varying_var] == value]
        if not subset.empty:
            plt.plot(
                subset[x_var], 
                subset[y_var], 
                marker='o', 
                label=f"{varying_var}={value:.3f}", 
                color=cmap(norm(value)),
            )
    
    # Title and labels
    title = "Averaged Over: " + ", ".join(averaging_vars)
    plt.title(f"Analysis of {x_var} vs {y_var}with Varying {varying_var}\n{title}")
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.legend(title=varying_var)
    plt.grid()
    plt.show()


def plot_analysis(df, fixed_vars, x_var):
    """
    Plots P_AD on the x-axis against a specified variable on the y-axis, fixing other variables.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        fixed_vars (dict): Dictionary of variables to fix with their values, e.g., {'alpha': 0.05, 'lambda': 0.05}.
        x_var (str): The variable to plot on the y-axis.
    """
    # Validate inputs
    if 'P_AD' not in df.columns:
        raise ValueError("The dataframe must contain the column 'P_AD' for x-axis plotting.")
    if x_var not in df.columns:
        raise ValueError(f"The specified y-axis variable '{x_var}' is not in the dataframe.")
    for var in fixed_vars:
        if var not in df.columns:
            raise ValueError(f"The fixed variable '{var}' is not in the dataframe.")
    
    # Filter the dataframe based on fixed variables
    filtered_df = df.copy()
    for var, value in fixed_vars.items():
        filtered_df = filtered_df[filtered_df[var] == value]
    
    # Check if there is any data left after filtering
    if filtered_df.empty:
        print(f"No data available for the specified fixed variables: {fixed_vars}")
        return
    
    # Plot P_AD on the x-axis and x_var on the y-axis
    plt.figure(figsize=(10, 6))
    plt.plot(
        filtered_df[x_var], 
        filtered_df['P_AD'], 
        marker='o', 
        label=f"{x_var} vs P_AD"
    )
    # Title and labels
    title = "Fixed Parameters: " + ", ".join(f"{key}={value}" for key, value in fixed_vars.items())
    plt.title(f"Analysis of {x_var} vs P_AD\n{title}")
    plt.xlabel(x_var)
    plt.ylabel('P_AD')
    plt.legend()
    plt.grid()
    plt.show()


def heatmap_mean(df, yaxis, xaxis, value='P_AD'):
    heatmap_data = df.pivot_table(index=yaxis, columns=xaxis, values=value, aggfunc='mean')

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        cmap='YlGnBu',
        annot=False,
        fmt='.2f',
        cbar_kws={'label': r'Mean $P_{AD}$'}
    )

    # Title with fixed LaTeX-style P_{AD}
    title_str = r'Heatmap of $P_{AD}$ by ' + f'{yaxis} and {xaxis}\n(mean of all other parameters)'
    plt.title(title_str, fontsize=16)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.gca().invert_yaxis()
    plt.show()

def convert_dict_to_mean_df(data_dict, mean = True):
    dfs = []
    for key in data_dict.keys():
        qvalues = data_dict[key].copy()
        flattened_qvalues = [array.flatten() for array in qvalues]
        stacked_qvalues = np.vstack(flattened_qvalues)
        df = pd.DataFrame(stacked_qvalues, columns=["Q1", "Q2", "Q3", "Q4"])
        dfs.append(df)
    

    if mean:
        max_length = max(len(df) for df in dfs)

        mean_rows = []

        for i in range(max_length):
            row_means = []

            for df in dfs:
                if i < len(df):  # If the row exists in this DataFrame
                    row_means.append(df.iloc[i][['Q1', 'Q2', 'Q3', 'Q4']].values)
                else:  # If the row does not exist, append NaNs
                    row_means.append([np.nan, np.nan, np.nan, np.nan])

            # Compute the mean for each column (Q1, Q2, Q3, Q4) across all DataFrames
            mean_row = np.nanmean(row_means, axis=0)  # np.nanmean ignores NaN values

            mean_rows.append(mean_row)

        # Convert the list of row-wise means into a DataFrame
        mean_df = pd.DataFrame(mean_rows, columns=['Q1', 'Q2', 'Q3', 'Q4'])
        return mean_df
    
    else:
        return dfs


def generate_unique_2x2():
    # Generate a 2x2 array with random values
    array = np.random.rand(2, 2)
    
    # Ensure the values are distinct when rounded to two decimal points
    while len(np.unique(array.round(1))) < 4:
        array = np.random.rand(2, 2)
    
    return array

def generate_unique_complete_samplings(T, delta):
    """
    Generate a 2x2 array with random values in the range [0, T / (1 - delta)]
    ensuring that all values are distinct when rounded to one decimal place.
    
    Parameters:
        T (float): Upper limit factor.
        delta (float): Factor to adjust the range.
        
    Returns:
        np.ndarray: A 2x2 array with unique rounded values.
    """
    # Calculate the upper limit
    upper_limit = T / (1 - delta)
    
    # Generate a 2x2 array with random values in the range [0, upper_limit]
    array = np.random.uniform(0, upper_limit, (2, 2))
    
    # Ensure the values are distinct when rounded to one decimal place
    while len(np.unique(array.round(1))) < 4:
        array = np.random.uniform(0, upper_limit, (2, 2))
    
    return array

def latin_hypercube_sampling(T, delta):
    
    upper_limit = T / (1 - delta)
    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=4)
    unique_2x2_array = sample.reshape(2, 2)
    while len(np.unique(unique_2x2_array)) < 4:
        sample = sampler.random(n=1)
        unique_2x2_array = sample.reshape(2, 2)
    
    return unique_2x2_array

def create_line(value, length=10):
    """
    Create a line of repeated values.

    Parameters:
        value (float): The value to repeat.
        length (int): The number of times to repeat the value.

    Returns:
        list: A list containing the repeated values.
    """
    return [value for _ in range(length)]


def plot_analysis_multiple_lines_sub(df, fixed_vars, x_var, varying_var, ax, show_legend, y_limits=None):
    """
    Plots multiple lines of P_AD against a specified variable on the x-axis, 
    with each line representing a unique value of the varying variable.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        fixed_vars (dict): Dictionary of variables to fix with their values, e.g., {'delta': 0.786, 'lambda': 0.132}.
        x_var (str): The variable to plot on the x-axis.
        varying_var (str): The variable to vary, creating multiple lines.
        ax (matplotlib.axes.Axes): The axis to plot on.
        show_legend (bool): Flag to display the legend on the first subplot.
        y_limits (tuple, optional): Tuple of (y_min, y_max) for the y-axis limits.
    """
    # Validate inputs
    if 'P_AD' not in df.columns:
        raise ValueError("The dataframe must contain the column 'P_AD' for plotting.")
    if x_var not in df.columns:
        raise ValueError(f"The specified x-axis variable '{x_var}' is not in the dataframe.")
    if varying_var not in df.columns:
        raise ValueError(f"The specified varying variable '{varying_var}' is not in the dataframe.")
    for var in fixed_vars:
        if var not in df.columns:
            raise ValueError(f"The fixed variable '{var}' is not in the dataframe.")
    
    # Filter the dataframe based on fixed variables
    filtered_df = df.copy()
    for var, value in fixed_vars.items():
        filtered_df = filtered_df[filtered_df[var] == value]
    
    # Check if there is any data left after filtering
    if filtered_df.empty:
        print(f"No data available for the specified fixed variables: {fixed_vars}")
        return
    
    # Get unique values of the varying variable
    unique_values = sorted(filtered_df[varying_var].unique())    
    # Normalize unique values for colormap
    norm = plt.Normalize(min(unique_values), max(unique_values))
    cmap = cm.get_cmap('viridis')
    
    # Plot each unique value as a separate line
    for value in unique_values:
        subset = filtered_df[filtered_df[varying_var] == value]
        if not subset.empty:
            ax.plot(
                subset[x_var], 
                subset['P_GT'], 
                marker='o', 
                label=f"{varying_var}={value:.3f}", 
                color=cmap(norm(value)),
            )
    
    # Title and labels
    title = "Fixed Parameters: " + ", ".join(f"{key}={value}" for key, value in fixed_vars.items())
    ax.set_title(f"Analysis of {x_var} vs P_GT with Varying {varying_var}\n{title}")
    ax.set_xlabel(x_var)
    ax.set_ylabel('P_GT')
    if show_legend:
        ax.legend(title=varying_var)
    ax.grid()

    # Set y-limits if provided
    if y_limits is not None:
        ax.set_ylim(y_limits)


def trajectory_dinamics_plot(x, y, zoom_range=None):
    """
    zoom_range = {'xlim': (x_min, x_max), 'ylim': (y_min, y_max)}
    """
    u = x[1:] - x[:-1]
    v = y[1:] - y[:-1]

    plt.figure(figsize=(10, 10))
    plt.quiver(x[:-1], y[:-1], u, v, angles='xy', scale_units='xy', scale=1, color='grey')
    plt.scatter(x, y, color='black', s=10)
    plt.title("Trajectory Dynamics")
    plt.grid(True)

    # Apply zoom if specified
    if zoom_range is not None:
        if 'xlim' in zoom_range:
            plt.xlim(zoom_range['xlim'])
        if 'ylim' in zoom_range:
            plt.ylim(zoom_range['ylim'])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def multi_centered_trajectory_plot(*trajectories, labels=None, colors=None):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 8))

    # Set default colors if not provided
    default_colors = ['crimson', 'teal', 'orange', 'blue', 'green', 'magenta', 'brown', 'slategray']
    if colors is None:
        colors = default_colors
    if labels is None:
        labels = [f'Trajectory {i+1}' for i in range(len(trajectories))]

    # Determine max range for auto-scaling
    all_x = np.concatenate([traj[0] for traj in trajectories])
    all_y = np.concatenate([traj[1] for traj in trajectories])
    max_val = max(np.max(np.abs(all_x)), np.max(np.abs(all_y)))
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Grid and axes setup
    ax.set_xticks(np.arange(-max_val*1.1, max_val*1.1, 1))
    ax.set_yticks(np.arange(-max_val*1.1, max_val*1.1, 1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.1)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)

    # Hide spines
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    # Axis arrows
    #ax.annotate('', xy=(1, 0), xytext=(0, 0),arrowprops=dict(facecolor='purple', width=1.5, headwidth=8))
    #ax.annotate('', xy=(0, 1), xytext=(0, 0),arrowprops=dict(facecolor='purple', width=1.5, headwidth=8))

    # Plot each trajectory
    for i, (x, y) in enumerate(trajectories):
        u = x[1:] - x[:-1]
        v = y[1:] - y[:-1]
        color = colors[i % len(colors)]
        label = labels[i] if i < len(labels) else f'Trajectory {i+1}'
        
        ax.quiver(x[:-1], y[:-1], u, v, angles='xy', scale_units='xy', scale=1,
                  color=color, width=0.003, headwidth=4, headlength=6, label=label)
        ax.scatter(x, y, color=color, s=10)

    ax.set_aspect('equal')
    ax.set_xlabel('x', labelpad=10, fontsize=12)
    ax.set_ylabel('y', labelpad=10, fontsize=12)
    ax.set_title("Trajectories on Cartesian Grid", fontsize=14)
    ax.legend()
    plt.show()