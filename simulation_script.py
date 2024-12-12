# For tracking progress
M_value = 100
alpha_values = [0.05,0.1,0.2]
lambda_values = [0.05,0.1,0.2]
delta_values = [0.5,0.6,0.7]  # 1000 equally spaced values from 0 to 0.999
epsilon_values = [0.05,0.1,0.2]  # 1000 equally spaced values from 0 to 0.99
random_uniform_values = [False]

def jaajo
to continue tomorrow
    results = []

    # Generate all parameter combinations
    parameter_combinations = list(itertools.product(alpha_values, lambda_values, epsilon_values, delta_values, random_uniform_values))

    # Iterate through each parameter combination with tqdm progress bar
    for alpha, lambda_val, e, delta, random_uniform in tqdm(parameter_combinations, desc="Processing combinations"):
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

        # Optionally, set 'Index' as the DataFrame index (if you need it)
        df_merged.set_index('Index', inplace=True)

        # Count how many times Strategy_1 and Strategy_2 are different
        count_diff = (df_merged['Strategy_1'] != df_merged['Strategy_2']).sum()

        # Count how many times both Strategy_1 and Strategy_2 have the value 1
        P_AD = ((df_merged['Strategy_1'] == 1) & (df_merged['Strategy_2'] == 1)).sum() / M_value

        # Count how many times both Strategy_1 and Strategy_2 have the value 2
        P_GT = ((df_merged['Strategy_1'] == 2) & (df_merged['Strategy_2'] == 2)).sum() / M_value

        # Store results for this parameter combination
        results.append({
            'alpha': alpha,
            'lambda': lambda_val,
            'e': e,
            'delta': delta,
            'random_uniform': random_uniform,
            'P_AD': P_AD,
            'P_GT': P_GT,
            'nonconv': count_diff
        })

# Optionally, save results to a file
results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)

if __name__ == '__main__':

