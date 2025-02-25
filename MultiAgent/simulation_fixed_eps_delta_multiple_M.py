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

    # Save results to file
    results_df = pd.DataFrame(results)
    results_df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
    print(f"Saved {len(results)} results to {file_path}.")
    
    # Clear memory
    del results
    gc.collect()

if __name__ == "__main__":
    alpha_values = np.linspace(0.01, 0.5, 3)
    lambda_values = np.linspace(0.01, 0.99, 25)
    M_values = [20, 100, 500, 1000, 5000]
    
    epsilon_delta_pairs = [
        (0.05, 0.95), (0.35, 0.95), (0.6, 0.95),
        (0.05, 0.75), (0.35, 0.75), (0.6, 0.75),
        (0.05, 0.5), (0.35, 0.5), (0.6, 0.5),
        (0.05, 0.35), (0.35, 0.35), (0.6, 0.35)
    ]
    
    parameter_combinations = list(itertools.product(M_values, alpha_values, lambda_values, [e for e, d in epsilon_delta_pairs], [d for e, d in epsilon_delta_pairs]))
    
    print(f"Generated {len(parameter_combinations)} combinations.")
    run_experiment(parameter_combinations)
