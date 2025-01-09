import itertools
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from MultiAgent_DQ import Algorithm
from utils import convert_dict_to_mean_df

def run_single_simulation(params):
    """
    Function to run a single simulation.
    """
    alpha, lambda_val, epsilon, delta, random_uniform, M_value = params

    # Initialize and run the algorithm
    algo = Algorithm(e=epsilon, lambda_val=lambda_val, 
                     alpha=alpha, delta=delta,
                     random_uniform=random_uniform, 
                     tolerance=1e-5, M=M_value, verbose=False)
    q1_history, q2_history = algo.run()

    # Process results to calculate P_AD, P_GT, etc.
    dfs_1 = convert_dict_to_mean_df(q1_history, mean=False)
    dfs_2 = convert_dict_to_mean_df(q2_history, mean=False)

    strategy_played_1, strategy_played_2 = {}, {}
    for i in range(len(dfs_1)):
        last_1 = dfs_1[i].iloc[-1]
        last_2 = dfs_2[i].iloc[-1]

        GT_1 = (last_1['Q1'] > last_1['Q3']) & (last_1['Q2'] < last_1['Q4'])
        AD_1 = (last_1['Q1'] < last_1['Q3']) & (last_1['Q2'] < last_1['Q4'])
        AC_1 = (last_1['Q1'] > last_1['Q3']) & (last_1['Q2'] > last_1['Q4'])

        GT_2 = (last_2['Q1'] > last_2['Q3']) & (last_2['Q2'] < last_2['Q4'])
        AD_2 = (last_2['Q1'] < last_2['Q3']) & (last_2['Q2'] < last_2['Q4'])
        AC_2 = (last_2['Q1'] > last_2['Q3']) & (last_2['Q2'] > last_2['Q4'])

        strategy_played_1[i] = 1 if AD_1 else 2 if GT_1 else 3 if AC_1 else 0
        strategy_played_2[i] = 1 if AD_2 else 2 if GT_2 else 3 if AC_2 else 0

    df_strategy_1 = pd.DataFrame(list(strategy_played_1.items()), columns=['Index', 'Strategy_1'])
    df_strategy_2 = pd.DataFrame(list(strategy_played_2.items()), columns=['Index', 'Strategy_2'])
    df_merged = pd.merge(df_strategy_1, df_strategy_2, on='Index')

    count_diff = (df_merged['Strategy_1'] != df_merged['Strategy_2']).sum()
    P_AD = ((df_merged['Strategy_1'] == 1) & (df_merged['Strategy_2'] == 1)).sum() / M_value
    P_GT = ((df_merged['Strategy_1'] == 2) & (df_merged['Strategy_2'] == 2)).sum() / M_value

    return {
        'alpha': alpha,
        'lambda': lambda_val,
        'epsilon': epsilon,
        'delta': delta,
        'random_uniform': random_uniform,
        'P_AD': P_AD,
        'P_GT': P_GT,
        'nonconv': count_diff
    }

def run_simulations(alpha_values, lambda_values, epsilon_values, delta_values, random_uniform_values):
    """
    Function to run simulations across parameter combinations.
    """
    M_value = 100 #reduce
    parameter_combinations = itertools.product(
        alpha_values, lambda_values, epsilon_values, delta_values, random_uniform_values, [M_value]
    )
    
    # Write results incrementally to CSV
    with Pool(processes=max(1, cpu_count() // 4)) as pool, open("results.csv", "w") as f:
        print(f"Using {max(1, cpu_count() // 4)} CPU cores.")
        print("Running simulations...")
        
        # Write the CSV header
        header = "alpha,lambda,epsilon,delta,random_uniform,P_AD,P_GT,nonconv\n"
        f.write(header)

        # Use imap for progress tracking
        total_combinations = np.prod([len(alpha_values), len(lambda_values), 
                                       len(epsilon_values), len(delta_values), 
                                       len(random_uniform_values)])
        for idx, result in enumerate(pool.imap(run_single_simulation, parameter_combinations, chunksize=10), 1):
            line = f"{result['alpha']},{result['lambda']},{result['epsilon']},{result['delta']},{result['random_uniform']},{result['P_AD']},{result['P_GT']},{result['nonconv']}\n"
            f.write(line)

            # Print simple progress
            if idx % (total_combinations // 10) == 0 or idx == total_combinations:
                progress_percentage = (idx / total_combinations) * 100
                print(f"Progress: {progress_percentage:.1f}% ({idx}/{total_combinations})")

    print("Simulation completed. Results saved to 'results.csv'.")


def run_simulations_save_incr(alpha_values, lambda_values, epsilon_values, delta_values, random_uniform_values):
    """
    Function to run simulations across parameter combinations and save results incrementally.
    """
    import itertools
    from multiprocessing import cpu_count, Pool

    M_value = 100
    parameter_combinations = list(itertools.product(alpha_values, lambda_values, epsilon_values, delta_values, random_uniform_values, [M_value]))
    total_combinations = len(parameter_combinations)

    print(f"Total parameter combinations: {total_combinations}")

    # Open the file once to write results incrementally
    with open("results.csv", "w") as f:
        print(f"Using {max(1, cpu_count() // 4)} CPU cores.")
        
        # Write the CSV header
        header = "alpha,lambda,epsilon,delta,random_uniform,P_AD,P_GT,nonconv\n"
        f.write(header)

        # Use a pool for parallel processing
        with Pool(processes=max(1, cpu_count() // 4)) as pool:
            chunksize = 10
            for idx, result in enumerate(pool.imap(run_single_simulation, parameter_combinations, chunksize=chunksize), 1):
                # Write the result to the file immediately
                line = f"{result['alpha']},{result['lambda']},{result['epsilon']},{result['delta']},{result['random_uniform']},{result['P_AD']},{result['P_GT']},{result['nonconv']}\n"
                f.write(line)
                f.flush()  # Ensure the data is written to disk immediately

                # Simple progress print
                if idx % (total_combinations // 10) == 0 or idx == total_combinations:
                    progress_percentage = (idx / total_combinations) * 100
                    print(f"Progress: {progress_percentage:.1f}% ({idx}/{total_combinations})")

    print("Simulation completed. Results saved to 'results.csv'.")

if __name__ == '__main__':
    # Define parameter ranges
    alpha_values = np.linspace(0.01, 1, 100)  # 100 values between 0 and 1
    lambda_values = np.linspace(0.01, 0.995, 100)
    delta_values = np.linspace(0.01, 0.995, 100) #zoom in here with a timelimit higher 
    epsilon_values = np.linspace(0.01, 0.995, 100)
    random_uniform_values = [False]

    run_simulations_save_incr(alpha_values, lambda_values, epsilon_values, delta_values, random_uniform_values)
