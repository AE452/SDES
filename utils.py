import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import Normalize
import plotly.graph_objects as go
from matplotlib import colormaps
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib as mpl

# Use LaTeX for text rendering
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def plot_multiple_panels(df, param_sets, x_var, varying_var, y_var, m, use_same_scale=True):
    """
    Plots multiple subplots with a single shared legend.
    """

    # Mapping of variable names to Greek labels
    greek_labels = {
        "alpha": r"$\alpha$",
        "beta": r"$\beta$",
        "delta": r"$\delta$",
        "lambda": r"$\lambda$",
        "mu": r"$\mu$",
        "M": r"$M$",
        "P_GT": r"$P_{GT}$",
        "P_AD": r"$P_{AD}$",
    }

    if y_var not in df.columns:
        raise ValueError(f"The specified y-axis variable '{y_var}' is not in the dataframe.")

    # Calculate global y-limits from all data if use_same_scale is True
    if use_same_scale:
        all_y_values = []
        for fixed_vars in param_sets:
            fixed_vars_copy = fixed_vars.copy()
            fixed_vars_copy['M'] = m
            filtered_df = df.copy()
            for var, value in fixed_vars_copy.items():
                filtered_df = filtered_df[filtered_df[var] == value]
            if not filtered_df.empty:
                all_y_values.extend(filtered_df[y_var].values)
        
        if all_y_values:
            y_min = min(all_y_values)
            y_max = max(all_y_values)
            # Add 5% padding on each side for better visualization
            y_range = y_max - y_min
            padding = y_range * 0.05 if y_range > 0 else 0.1
            y_limits = (y_min - padding, y_max + padding)
        else:
            y_limits = (-0.1, 0.7)  # Fallback if no data
    else:
        y_limits = None
    
    wspace, hspace = (0.02, 0.15) if use_same_scale else (0.2, 0.2)

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), 
                             gridspec_kw={'wspace': wspace, 'hspace': hspace}, 
                             constrained_layout=False)
    axes = axes.flatten()
    
    handles, labels = None, None

    for i, fixed_vars in enumerate(param_sets):
        fixed_vars['M'] = m
        handles, labels = plot_analysis_multiple_lines_sub(
            df, fixed_vars, x_var, varying_var, y_var, axes[i], y_limits
        )
        # Set axis labels per subplot
        # Only show x-label for bottom row
        if i in [4,5]:
            axes[i].set_xlabel(greek_labels.get(x_var, x_var))
        else:
            axes[i].set_xlabel("")
            axes[i].set_xticklabels([])

        axes[i].set_ylabel(greek_labels.get(y_var, y_var))

    # Add a single legend outside the plot
    if handles and labels:
        fig.legend(handles, labels, title=greek_labels.get(varying_var, varying_var), 
                   loc='upper right', bbox_to_anchor=(1.1, 1))

    # Hide redundant labels for shared scales
    if use_same_scale:
        for ax in axes:
            ax.label_outer()

    plt.show()
def plot_analysis_multiple_lines_sub(df, fixed_vars, x_var, varying_var, y_var, ax, y_limits=None):
    """
    Plots multiple lines in a single subplot without adding a legend.
    """

    greek_labels = {
        "alpha": r"$\alpha$",
        "beta": r"$\beta$",
        "delta": r"$\delta$",
        "epsilon": r"$\varepsilon$",
        "e": r"$\varepsilon$",

        "lambda": r"$\lambda$",
        "mu": r"$\mu$",
        "M": r"$M$",
        "P_GT": r"$P_{GT}$",
        "P_AD": r"$P_{AD}$",
    }

    if y_var not in df.columns:
        raise ValueError(f"The dataframe must contain the column '{y_var}' for plotting.")

    filtered_df = df.copy()
    for var, value in fixed_vars.items():
        filtered_df = filtered_df[filtered_df[var] == value]

    if filtered_df.empty:
        return None, None

    unique_values = sorted(filtered_df[varying_var].unique())    
    norm = plt.Normalize(min(unique_values), max(unique_values))
    cmap = colormaps.get_cmap('viridis')

    handles, labels = [], []

    for value in unique_values:
        subset = filtered_df[filtered_df[varying_var] == value]
        if not subset.empty:
            label = rf"{greek_labels.get(varying_var, varying_var)} = {value:.3f}"
            line, = ax.plot(
                subset[x_var], subset[y_var], marker='o',
                label=label, color=cmap(norm(value))
            )
            handles.append(line)
            labels.append(label)

    # Format subplot title with Greek letters
    title_parts = [
        rf"{greek_labels.get(k, k)}={v}"
        for k, v in fixed_vars.items() if k != 'M'
    ]
    ax.set_title(r"Fixed: " + ", ".join(title_parts))

    ax.grid()

    if y_limits:
        ax.set_ylim(y_limits)

    return handles, labels



def interactive_3d_voxel(df, xaxis, yaxis, zaxis, value='P_AD', html=False):
    greek_labels = {
    "alpha": "α",
    "beta": "β",
    "delta": "δ",
    "epsilon": "ε",
    "e": "ε",
    "lambda": "λ",
    "mu": "μ",
    "M": "M",  # Latin letter
    "P_GT": "P_GT",  # Optional: use subscript formatting if needed
    "P_AD": "P_AD",
}


    grouped = df.groupby([xaxis, yaxis, zaxis])[value].mean().reset_index()

    # Map categorical to numeric
    x_labels = sorted(df[xaxis].unique())
    y_labels = sorted(df[yaxis].unique())
    z_labels = sorted(df[zaxis].unique())

    x_map = {k: i for i, k in enumerate(x_labels)}
    y_map = {k: i for i, k in enumerate(y_labels)}
    z_map = {k: i for i, k in enumerate(z_labels)}

    # Map to numeric positions
    grouped[str(xaxis)] = grouped[xaxis].map(x_map)
    grouped[str(yaxis)] = grouped[yaxis].map(y_map)
    grouped[str(zaxis)] = grouped[zaxis].map(z_map)

    fig = go.Figure(data=go.Scatter3d(
        x=grouped[str(xaxis)],
        y=grouped[str(yaxis)],
        z=grouped[str(zaxis)],
        mode='markers',
        marker=dict(
            size=20,
            color=grouped[value],
            colorscale='Inferno',
            colorbar=dict(title=greek_labels.get(value, value)),
            opacity=0.8,
        ),
        text=[f'{value}: {v:.2f}' for v in grouped[value]]
    ))

    # ✅ Reduce tick labels (every 2nd for clarity)
    x_ticks_sparse = x_labels[::2]
    y_ticks_sparse = y_labels[::2]
    z_ticks_sparse = z_labels[::2]

    fig.update_layout(
        width=1000,
        height=800,
        scene=dict(
            xaxis=dict(
                title=greek_labels.get(xaxis, xaxis),
                tickvals=[x_map[v] for v in x_ticks_sparse],
                ticktext=x_ticks_sparse
            ),
            yaxis=dict(
                title=greek_labels.get(yaxis, yaxis),
                tickvals=[y_map[v] for v in y_ticks_sparse],
                ticktext=y_ticks_sparse
            ),
            zaxis=dict(
                title=greek_labels.get(zaxis, zaxis),
                tickvals=[z_map[v] for v in z_ticks_sparse],
                ticktext=z_ticks_sparse
            ),
        ),
        title=f"Interactive 3D Heatmap of {greek_labels.get(value, value)}<br>"
              f"vs {greek_labels.get(xaxis, xaxis)}, {greek_labels.get(yaxis, yaxis)}, {greek_labels.get(zaxis, zaxis)}",
        margin=dict(l=0, r=0, b=0, t=100)
    )

    if html:
        fig.write_html("3dinteractivea.html", full_html=True, include_plotlyjs='cdn')
    else: 
        fig.show()




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
    cmap = colormaps.get_cmap('viridis')
    
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
    cmap = colormaps.get_cmap('viridis')
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
    greek_labels = {
        "alpha": r"$\alpha$",
        "beta": r"$\beta$",
        "delta": r"$\delta$",
        "epsilon": r"$\varepsilon$",
        "e": r"$\varepsilon$",
        "lambda": r"$\lambda$",
        "mu": r"$\mu$",
        "M": r"$M$",
        "P_GT": r"$P_{GT}$",
        "P_AD": r"$P_{AD}$",
    }

    heatmap_data = df.pivot_table(index=yaxis, columns=xaxis, values=value, aggfunc='mean')

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        cmap='YlGnBu',
        annot=False,
        fmt='.2f',
        cbar_kws={'label': rf"Mean {greek_labels.get(value, value)}"}
    )

    # Title using Greek symbols
    title_str = rf"Heatmap of {greek_labels.get(value, value)} by " \
                rf"{greek_labels.get(yaxis, yaxis)} and {greek_labels.get(xaxis, xaxis)}" \
                "\n(mean of all other parameters)"
    plt.title(title_str, fontsize=16)

    plt.xlabel(greek_labels.get(xaxis, xaxis))
    plt.ylabel(greek_labels.get(yaxis, yaxis))
    ax = plt.gca()
    yticks = ax.get_yticks()
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::2])  # Show every second x-t
    ax.set_yticks(yticks[::2]) 

    ax.invert_yaxis()
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

def heatmap_fixed_pairs(df, yaxis, xaxis, value, fixed_pairs):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid for subplots
    axes = axes.flatten()  # Flatten the array for easy indexing

    for i, (alpha_val, lambda_val) in enumerate(fixed_pairs):
        ax = axes[i]

        # Filter data for the specific (alpha, lambda) pair
        filtered_df = df[(df['alpha'] == alpha_val) & (df['lambda'] == lambda_val)]

        if filtered_df.empty:
            ax.set_title(f'No data for alpha={alpha_val}, lambda={lambda_val}')
            ax.axis('off')  # Hide axes if no data
            continue

        # Create the heatmap
        heatmap_data = filtered_df.pivot_table(index=yaxis, columns=xaxis, values=value, aggfunc='mean')
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=False, fmt='.2f', 
                    cbar_kws={'label': f'Mean {value}'}, ax=ax)

        # Titles and labels
        ax.set_title(f'alpha={alpha_val}, lambda={lambda_val}')
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.show()



def heatmap_3d_voxels(df, xaxis, yaxis, zaxis, value='P_AD'):
    # Compute mean value for each (x, y, z) triple
    grouped = df.groupby([xaxis, yaxis, zaxis])[value].mean().reset_index()

    # Map categories to integer indices
    x_labels = sorted(df[xaxis].unique())
    y_labels = sorted(df[yaxis].unique())
    z_labels = sorted(df[zaxis].unique())

    x_map = {k: i for i, k in enumerate(x_labels)}
    y_map = {k: i for i, k in enumerate(y_labels)}
    z_map = {k: i for i, k in enumerate(z_labels)}

    # Prepare empty 3D grid
    shape = (len(x_labels), len(y_labels), len(z_labels))
    grid = np.full(shape, np.nan)

    for _, row in grouped.iterrows():
        xi = x_map[row[xaxis]]
        yi = y_map[row[yaxis]]
        zi = z_map[row[zaxis]]
        grid[xi, yi, zi] = row[value]

    # Create voxel mask
    mask = ~np.isnan(grid)

    # Normalize values for colormap
    norm = Normalize(vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    cmap = colormaps.get_cmap('viridis')

    # Build RGBA colors based on value
    colors = np.zeros(grid.shape + (4,))
    for idx, val in np.ndenumerate(grid):
        if not np.isnan(val):
            colors[idx] = cmap(norm(val))

    # Plot voxels
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(mask, facecolors=colors, edgecolor='k')

    ax.set_xticks(np.arange(len(x_labels)) + 0.5)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(y_labels)) + 0.5)
    ax.set_yticklabels(y_labels, rotation=45, ha='right')
    ax.set_zticks(np.arange(len(z_labels)) + 0.5)
    ax.set_zticklabels(z_labels)

    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    ax.set_zlabel(zaxis)
    ax.set_title(f'3D Voxel Heatmap of {value} by {xaxis}, {yaxis}, and {zaxis}')

    # Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(f'Mean {value}')

    plt.tight_layout()
    plt.show()

def trajectory_dynamics_plot(x, y, zoom_range=None):
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