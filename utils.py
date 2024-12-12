import pandas as pd
import numpy as np

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
