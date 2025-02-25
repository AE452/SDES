from MultiAgent_DQ import *
from utils import *
import pandas as pd  
import numpy as np
import itertools
from multiprocessing import Pool, cpu_count


def run_single_simulation(params):
    """
    Run the simulation for a single parameter combination.
    """
    alpha, lambda_val, e, delta, random_uniform, M_value = params

    # Initialize the algorithm with the current parameter combination
    algo = Algorithm(e=e, lambda_val=lambda_val, 
                     alpha=alpha, delta=delta,
                     random_uniform=random_uniform, tolerance=1e-5, 
                     M=M_value, verbose=False)

    # Run the algorithm
    q1_history_per_sample, q2_history_per_sample = algo.run()

    # Convert results to dataframes
    dfs_1 = convert_dict_to_mean_df(q1_history_per_sample, mean=False)
    dfs_2 = convert_dict_to_mean_df(q2_history_per_sample, mean=False)

    # Initialize strategies played
    strategy_played_1 = {}
    strategy_played_2 = {}

    # Iterate over the dataframe rows
    for i in range(len(dfs_1)):
        last_1 = dfs_1[i].iloc[-1]  # Get the last row using iloc
        last_2 = dfs_2[i].iloc[-1]

        # Define conditions for both players
        GT_1 = (last_1['Q1'] > last_1['Q3']) & (last_1['Q2'] < last_1['Q4'])
        AD_1 = (last_1['Q1'] < last_1['Q3']) & (last_1['Q2'] < last_1['Q4'])
        AC_1 = (last_1['Q1'] > last_1['Q3']) & (last_1['Q2'] > last_1['Q4'])

        GT_2 = (last_2['Q1'] > last_2['Q3']) & (last_2['Q2'] < last_2['Q4'])
        AD_2 = (last_2['Q1'] < last_2['Q3']) & (last_2['Q2'] < last_2['Q4'])
        AC_2 = (last_2['Q1'] > last_2['Q3']) & (last_2['Q2'] > last_2['Q4'])

        # Assign strategy for both players
        strategy_map = {
            AD_1: 1,  # Always Defecting
            GT_1: 2,  # Grim Trigger
            AC_1: 3,  # Always Cooperating
        }

        for player, strategy in [(AD_1, strategy_map), (GT_1, strategy_map), (AC_1, strategy_map)]:
            if player:
                strategy_played_1[i] = strategy[player]
                break

        for player, strategy in [(AD_2, strategy_map), (GT_2, strategy_map), (AC_2, strategy_map)]:
            if player:
                strategy_played_2[i] = strategy[player]
                break

    # Convert strategy_played_1 and strategy_played_2 dictionaries to DataFrames
    df_strategy_1 = pd.DataFrame(list(strategy_played_1.items()), columns=['Index', 'Strategy_1'])
    df_strategy_2 = pd.DataFrame(list(strategy_played_2.items()), columns=['Index', 'Strategy_2'])

    # Merge the two DataFrames on 'Index'
    df_merged = pd.merge(df_strategy_1, df_strategy_2, on='Index')

    # Count how many times Strategy_1 and Strategy_2 are different
    count_diff = (df_merged['Strategy_1'] != df_merged['Strategy_2']).sum()

    # Count how many times both Strategy_1 and Strategy_2 have the value 1
    P_AD = ((df_merged['Strategy_1'] == 1) & (df_merged['Strategy_2'] == 1)).sum() / M_value

    # Count how many times both Strategy_1 and Strategy_2 have the value 2
    P_GT = ((df_merged['Strategy_1'] == 2) & (df_merged['Strategy_2'] == 2)).sum() / M_value

    # Return results for this parameter combination
    return {
        'alpha': alpha,
        'lambda': lambda_val,
        'e': e,
        'delta': delta,
        'random_uniform': random_uniform,
        'P_AD': P_AD,
        'P_GT': P_GT,
        'nonconv': count_diff
    }


def run_simulations(alpha_values, lambda_values, epsilon_values, delta_values, random_uniform_values, M_value):
    results = []

    # Generate all parameter combinations
    parameter_combinations = list(itertools.product(alpha_values, lambda_values, epsilon_values, delta_values, random_uniform_values, [M_value]))

    total_combinations = len(parameter_combinations)
    print(f"Total parameter combinations: {total_combinations}")

    # Use multiprocessing to parallelize
    with Pool(processes=cpu_count()) as pool:
        for idx, result in enumerate(pool.imap(run_single_simulation, parameter_combinations), 1):
            results.append(result)
            
            # Print simple progress every 10%
            if idx % (total_combinations // 10) == 0 or idx == total_combinations:
                progress_percentage = (idx / total_combinations) * 100
                print(f"Progress: {progress_percentage:.1f}% ({idx}/{total_combinations})")

    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)
    print("Simulation completed. Results saved to 'results.csv'.")


if __name__ == '__main__':
    alpha_values = np.linspace(0.01, 1, 500)  # 100 values between 0 and 1
    lambda_values = np.linspace(0.01, 0.995, 500)
    delta_values = np.linspace(0.01, 0.995, 500)
    epsilon_values = np.linspace(0.01, 0.995, 500)
    random_uniform_values = [False]
    M_value

    run_simulations(alpha_values, lambda_values, epsilon_values, delta_values, random_uniform_values, M_value)
