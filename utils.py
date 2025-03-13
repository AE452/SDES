import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
                subset['P_AD'], 
                marker='o', 
                label=f"{varying_var}={value:.3f}", 
                color=cmap(norm(value)),
            )
    
    # Title and labels
    title = "Fixed Parameters: " + ", ".join(f"{key}={value}" for key, value in fixed_vars.items())
    plt.title(f"Analysis of {x_var} vs P_AD with Varying {varying_var}\n{title}")
    plt.xlabel(x_var)
    plt.ylabel('P_AD')
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


def heatmap_mean(df, yaxis, xaxis, value):

    # Pivot the dataframe to create a matrix where 'alpha' and 'lambda' are the indices and columns
    heatmap_data = df.pivot_table(index=yaxis, columns=xaxis, values=value, aggfunc='mean')

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=False, fmt='.2f', cbar_kws={'label': f'Mean {str(value)}'})

    # Set titles and labels
    plt.title(f'Heatmap of {str(value)} by {str(yaxis)} and {str(xaxis)} (mean of all other params)')
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
