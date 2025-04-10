import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from MultiAgent_DQ import *
import os


def detect_strategy(df_last):
    return np.where(
        (df_last['Q1'] < df_last['Q3']) & (df_last['Q2'] < df_last['Q4']), 1,
        np.where(
            (df_last['Q1'] > df_last['Q3']) & (df_last['Q2'] < df_last['Q4']), 2,
            np.where(
                (df_last['Q1'] > df_last['Q3']) & (df_last['Q2'] > df_last['Q4']), 3,
                4
            )
        )
    )

def process_single_run(row, M_value=500, random_uniform=False):
    alpha = row['alpha']
    lambda_b = row['lambda']
    eps = row['e']
    delta = row['delta']

    algo = Algorithm(e=eps, lambda_val=lambda_b, alpha=alpha, delta=delta,
                     random_uniform=random_uniform, tolerance=1e-5,
                     M=int(M_value), verbose=False)
    q1_algo, q2_algo, batch_algo, timeout_out, timeout_vi = algo.run()

    algo_1 = Algorithm_1(e=eps, lambda_val=lambda_b, alpha=alpha, delta=delta,
                         random_uniform=random_uniform, tolerance=1e-5,
                         M=int(M_value), verbose=False)
    q1_algo1, q2_algo1, batch_algo1, timeout_out_1, timeout_vi_1 = algo_1.run()

    dfs_1_algo = convert_dict_to_mean_df(q1_algo, mean=False)
    dfs_2_algo = convert_dict_to_mean_df(q2_algo, mean=False)
    dfs_1_algo1 = convert_dict_to_mean_df(q1_algo1, mean=False)
    dfs_2_algo1 = convert_dict_to_mean_df(q2_algo1, mean=False)

    dfs_1_last_algo = pd.concat([df.iloc[[-1]] for df in dfs_1_algo])
    dfs_2_last_algo = pd.concat([df.iloc[[-1]] for df in dfs_2_algo])
    dfs_1_last_algo1 = pd.concat([df.iloc[[-1]] for df in dfs_1_algo1])
    dfs_2_last_algo1 = pd.concat([df.iloc[[-1]] for df in dfs_2_algo1])

    strategy_1_algo = detect_strategy(dfs_1_last_algo)
    strategy_2_algo = detect_strategy(dfs_2_last_algo)
    strategy_1_algo1 = detect_strategy(dfs_1_last_algo1)
    strategy_2_algo1 = detect_strategy(dfs_2_last_algo1)

    strategy_df_algo = pd.DataFrame({'Strategy_1': strategy_1_algo, 'Strategy_2': strategy_2_algo})
    strategy_df_algo1 = pd.DataFrame({'Strategy_1': strategy_1_algo1, 'Strategy_2': strategy_2_algo1})

    P_AD_algo = ((strategy_df_algo['Strategy_1'] == 1) & (strategy_df_algo['Strategy_2'] == 1)).mean()
    P_GT_algo = ((strategy_df_algo['Strategy_1'] == 2) & (strategy_df_algo['Strategy_2'] == 2)).mean()
    nonconv_algo = (strategy_df_algo['Strategy_1'] != strategy_df_algo['Strategy_2']).sum()

    P_AD_algo1 = ((strategy_df_algo1['Strategy_1'] == 1) & (strategy_df_algo1['Strategy_2'] == 1)).mean()
    P_GT_algo1 = ((strategy_df_algo1['Strategy_1'] == 2) & (strategy_df_algo1['Strategy_2'] == 2)).mean()
    nonconv_algo1 = (strategy_df_algo1['Strategy_1'] != strategy_df_algo1['Strategy_2']).sum()

    time_stats_algo = pd.DataFrame(list(batch_algo.items()), columns=["Batch", "Time"])["Time"]
    time_stats_algo1 = pd.DataFrame(list(batch_algo1.items()), columns=["Batch", "Time"])["Time"]
    
    time_max_algo = time_stats_algo.max()
    time_min_algo = time_stats_algo.min()
    time_mean_algo = time_stats_algo.mean()

    time_max_algo1 = time_stats_algo1.max()
    time_min_algo1 = time_stats_algo1.min()
    time_mean_algo1 = time_stats_algo1.mean()

    timeout_out_sum_algo = pd.DataFrame(list(timeout_out.items()), columns=["Batch", "Flag_out"])['Flag_out'].sum()
    timeout_vi_sum_algo = pd.DataFrame(list(timeout_vi.items()), columns=["Batch", "Flag_vi"])['Flag_vi'].sum()

    timeout_out_sum_algo1 = pd.DataFrame(list(timeout_out_1.items()), columns=["Batch", "Flag_out"])['Flag_out'].sum()
    timeout_vi_sum_algo1 = pd.DataFrame(list(timeout_vi_1.items()), columns=["Batch", "Flag_vi"])['Flag_vi'].sum()

    return [
        {
            'Algorithm': 'Algorithm',
            'M': M_value,
            'alpha': alpha, 'lambda': lambda_b, 'e': eps, 'delta': delta,
            'random_uniform': random_uniform, 'P_AD': P_AD_algo, 'P_GT': P_GT_algo,
            'nonconv': nonconv_algo, 'time_max': time_max_algo,
            'time_min': time_min_algo, 'time_mean': time_mean_algo,
            'timeout_out': timeout_out_sum_algo, 'timeout_vi': timeout_vi_sum_algo
        },
        {
            'Algorithm': 'Algorithm_1',
            'M': M_value,
            'alpha': alpha, 'lambda': lambda_b, 'e': eps, 'delta': delta,
            'random_uniform': random_uniform, 'P_AD': P_AD_algo1, 'P_GT': P_GT_algo1,
            'nonconv': nonconv_algo1, 'time_max': time_max_algo1,
            'time_min': time_min_algo1, 'time_mean': time_mean_algo1,
            'timeout_out': timeout_out_sum_algo1, 'timeout_vi': timeout_vi_sum_algo1
        }
    ]

def run_all(high_non_conv_df, n_jobs=-1):
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_run)(row) for _, row in high_non_conv_df.iterrows()
    )
    # Flatten the list of lists
    flat_results = [result for pair in all_results for result in pair]
    return pd.DataFrame(flat_results)

def run_all_incremental(high_non_conv_df, output_path="algorithm_results_comparison_non_conv.csv", n_jobs=1, overwrite=True):
    if overwrite and os.path.exists(output_path):
        os.remove(output_path)

    def wrapper(row):
        results = process_single_run(row)
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path))

    Parallel(n_jobs=n_jobs)(
        delayed(wrapper)(row) for _, row in high_non_conv_df.iterrows()
    )

    return pd.read_csv(output_path)


if __name__ == "__main__":
    high_non_conv_df = pd.read_csv('./high_non_conv_df.csv')
    #results_df = run_all(high_non_conv_df, n_jobs=4)
    results_df = run_all_incremental(high_non_conv_df, output_path="algorithm_results_comparison_non_conv.csv", n_jobs=4)
    results_df.to_csv("algorithm_results_comparison_non_conv.csv", index=False)
