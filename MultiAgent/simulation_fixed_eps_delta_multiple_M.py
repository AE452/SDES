import itertools
import time
from MultiAgent_DQ import Algorithm
import sys
import pandas as pd
import numpy as np
import os
import gc
sys.path.append('../')
from utils import *

def run_experiment(parameter_combinations, file_path="results_sequential.csv"):
    results = []
    
    for M_value, alpha, lambda_val, e, delta in parameter_combinations:
        print(f"Running experiment with M={M_value}, alpha={alpha}, lambda={lambda_val}, e={e}, delta={delta}")

        
        algo = Algorithm(
            e=e, lambda_val=lambda_val, alpha=alpha, delta=delta,
            random_uniform=False, tolerance=1e-5, 
            M=M_value, verbose=False
        )
        q1_history_per_sample, q2_history_per_sample = algo.run()

        # Process results
        dfs_1 = convert_dict_to_mean_df(q1_history_per_sample, mean=False)
        dfs_2 = convert_dict_to_mean_df(q2_history_per_sample, mean=False)

        # Combine last rows from all DataFrames
        dfs_1_last = pd.concat([df.iloc[[-1]] for df in dfs_1])
        dfs_2_last = pd.concat([df.iloc[[-1]] for df in dfs_2])

        # Assign strategies using vectorization
        strategy_1 = np.where(
            (dfs_1_last['Q1'] < dfs_1_last['Q3']) & (dfs_1_last['Q2'] < dfs_1_last['Q4']), 1, 
            np.where(
                (dfs_1_last['Q1'] > dfs_1_last['Q3']) & (dfs_1_last['Q2'] < dfs_1_last['Q4']), 2, 
                np.where(
                    (dfs_1_last['Q1'] > dfs_1_last['Q3']) & (dfs_1_last['Q2'] > dfs_1_last['Q4']), 3, 
                    np.nan
                )
            )
        )
        strategy_2 = np.where(
            (dfs_2_last['Q1'] < dfs_2_last['Q3']) & (dfs_2_last['Q2'] < dfs_2_last['Q4']), 1, 
            np.where(
                (dfs_2_last['Q1'] > dfs_2_last['Q3']) & (dfs_2_last['Q2'] < dfs_2_last['Q4']), 2, 
                np.where(
                    (dfs_2_last['Q1'] > dfs_2_last['Q3']) & (dfs_2_last['Q2'] > dfs_2_last['Q4']), 3, 
                    np.nan
                )
            )
        )

        # Create strategy DataFrame
        strategy_df = pd.DataFrame({'Strategy_1': strategy_1, 'Strategy_2': strategy_2})

        # Calculate probabilities and non-convergence count
        P_AD = ((strategy_df['Strategy_1'] == 1) & (strategy_df['Strategy_2'] == 1)).mean()
        P_GT = ((strategy_df['Strategy_1'] == 2) & (strategy_df['Strategy_2'] == 2)).mean()
        nonconv = (strategy_df['Strategy_1'] != strategy_df['Strategy_2']).sum()

        # Save result
        results.append({'M': int(M_value),
            'alpha': alpha, 'lambda': lambda_val, 'e': e, 'delta': delta,
            'P_AD': P_AD, 'P_GT': P_GT, 'nonconv': nonconv
        })
        # Append to CSV
        result_df = pd.DataFrame([{
            'M': int(M_value),
            'alpha': alpha, 'lambda': lambda_val, 'e': e, 'delta': delta,
            'P_AD': P_AD, 'P_GT': P_GT, 'nonconv': nonconv
        }])
        result_df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
        print(f"Saved result for M={M_value}, alpha={alpha}, lambda={lambda_val}, e={e}, delta={delta} to {file_path}.")
        
        # Force write to disk
        os.fsync(os.open(file_path, os.O_RDWR))

        # Memory cleanup
        del result_df
        gc.collect()


if __name__ == "__main__":
    alpha_values = [0.01, 0.051, 0.092, 0.134, 0.175, 0.216, 0.258, 0.299, 0.34, 0.381, 0.422, 0.464, 0.505, 0.546, 0.588, 0.629, 0.67, 0.711, 0.753, 0.794, 0.835, 0.876, 0.918, 0.959]
    lambda_values = [0.01, 0.051, 0.092, 0.134, 0.175, 0.216, 0.258, 0.299, 0.34, 0.381, 0.422, 0.464, 0.505, 0.546, 0.588, 0.629, 0.67, 0.711, 0.753, 0.794, 0.835, 0.876, 0.918, 0.959]
    M_values = [20, 100, 500, 1000, 5000]
    epsilon_values = [0.05, 0.35, 0.6]
    delta_values = [0.35, 0.5, 0.75, 0.95]    
    parameter_combinations = list(itertools.product(M_values, alpha_values, lambda_values, epsilon_values, delta_values))
    
    print(f"Generated {len(parameter_combinations)} combinations.")
    run_experiment(parameter_combinations)
