import itertools
import time
from MultiAgent_DQ import Algorithm
from utils import convert_dict_to_mean_df
import pandas as pd
import numpy as np
import os
import gc
import concurrent.futures
import multiprocessing
from gc import collect

#this is the function used
def run_experiment_chunk_vect(parameter_combinations_chunk):
    
    results_chunk = []

    for idx, (M_value, alpha, lambda_val, e, delta, random_uniform) in enumerate(parameter_combinations_chunk):
        algo = Algorithm(
            e=e, lambda_val=lambda_val, alpha=alpha, delta=delta,
            random_uniform=random_uniform, tolerance=1e-5, 
            M=int(M_value), verbose=False
        )
        q1_history_per_sample, q2_history_per_sample = algo.run()

        # Process results
        dfs_1 = convert_dict_to_mean_df(q1_history_per_sample, mean=False)
        dfs_2 = convert_dict_to_mean_df(q2_history_per_sample, mean=False)

        # Combine last rows from all DataFrames
        dfs_1_last = pd.concat([df.iloc[[-1]] for df in dfs_1])
        dfs_2_last = pd.concat([df.iloc[[-1]] for df in dfs_2])

        # Assign strategies using vectorization
        #you can futher optimize this part because the detection is the same for both the players,
        # create detection function with df in input and then this np.where nested is used, 
        #probably one case is missing and it is going to np.nan (check), rarely happens but it happens
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
        results_chunk.append({'M': int(M_value),
            'alpha': alpha, 'lambda': lambda_val, 'e': e, 'delta': delta,
            'random_uniform': random_uniform, 'P_AD': P_AD, 'P_GT': P_GT,
            'nonconv': nonconv
        })

    return results_chunk

def run_experiment_chunk(parameter_combinations_chunk):
    """Run the experiment for a chunk of parameter combinations."""
    results_chunk = []
    M_value = 50
    
    for idx, (alpha, lambda_val, e, delta, random_uniform) in enumerate(parameter_combinations_chunk):
        algo = Algorithm(
            e=e, lambda_val=lambda_val, alpha=alpha, delta=delta,
            random_uniform=random_uniform, tolerance=1e-5, 
            M=M_value, verbose=False
        )
        q1_history_per_sample, q2_history_per_sample = algo.run()

        # Process results
        dfs_1 = convert_dict_to_mean_df(q1_history_per_sample, mean=False)
        dfs_2 = convert_dict_to_mean_df(q2_history_per_sample, mean=False)

        # Determine strategies
        strategy_played_1 = {
            i: 1 if (row['Q1'] < row['Q3']) and (row['Q2'] < row['Q4']) else
               2 if (row['Q1'] > row['Q3']) and (row['Q2'] < row['Q4']) else
               3 for i, row in dfs_1[-1].iterrows()
        }
        strategy_played_2 = {
            i: 1 if (row['Q1'] < row['Q3']) and (row['Q2'] < row['Q4']) else
               2 if (row['Q1'] > row['Q3']) and (row['Q2'] < row['Q4']) else
               3 for i, row in dfs_2[-1].iterrows()
        }

        # Calculate probabilities
        strategy_df = pd.DataFrame({'Strategy_1': strategy_played_1, 'Strategy_2': strategy_played_2})
        P_AD = ((strategy_df['Strategy_1'] == 1) & (strategy_df['Strategy_2'] == 1)).mean()
        P_GT = ((strategy_df['Strategy_1'] == 2) & (strategy_df['Strategy_2'] == 2)).mean()
        nonconv = (strategy_df['Strategy_1'] != strategy_df['Strategy_2']).sum()

        # Save result
        results_chunk.append({
            'alpha': alpha, 'lambda': lambda_val, 'e': e, 'delta': delta,
            'random_uniform': random_uniform, 'P_AD': P_AD, 'P_GT': P_GT,
            'nonconv': nonconv
        })

    return results_chunk


def save_results(results_chunk, file_path):
    """Save the chunk of results to a CSV file."""
    file_exists = os.path.exists(file_path)
    results_df = pd.DataFrame(results_chunk)
    if not results_df.empty:
        results_df.to_csv(file_path, mode='a', header=not file_exists, index=False)
        print(f"Saved {len(results_chunk)} results to {file_path}.")
    else:
        print("No data to save.")
    
    # Clear memory
    del results_chunk
    collect()
def chunk_generator(iterator, chunk_size):
    """Yield successive chunks of data from an iterator."""
    chunk = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def run_experiment(parameter_combinations, file_path="results_vect3.csv"):
    """Run the experiment for all parameter combinations using limited CPU and memory."""
    M_value = 50
    chunk_size = max(1, len(parameter_combinations) // os.cpu_count())
    print(f'chunk_size is {chunk_size}')
    parameter_combinations_chunks = [
        parameter_combinations[i:i + chunk_size]
        for i in range(0, len(parameter_combinations), chunk_size)
    ]

    total_chunks = len(parameter_combinations_chunks)

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() // 3) as executor:
        futures = {
            executor.submit(run_experiment_chunk_vect, chunk): idx
            for idx, chunk in enumerate(parameter_combinations_chunks)
        }

        for completed, future in enumerate(concurrent.futures.as_completed(futures)):
            results_chunk = future.result()
            save_results(results_chunk, file_path)

            # Free memory
            del results_chunk
            gc.collect()

            progress = (completed + 1) / total_chunks * 100
            print(f"\rProgress: {progress:.2f}% ({completed + 1}/{total_chunks})", end='')

    print("\nProcessing complete.")


# Define the chunk processing function at the top level
def process_chunk(chunk):
    # Ensure we are generating the correct combinations from the chunk
    return list(itertools.product(*chunk))  # Generates the combinations for this chunk

def generate_combinations_parallel(alpha_values, lambda_values, epsilon_values, delta_values, random_uniform_values, M):
    """Generate parameter combinations using multiprocessing."""
    # Split parameters into chunks for parallel processing
    chunks = [
        (M, alpha_values, lambda_values, epsilon_values, delta_values, random_uniform_values)
    ]
    
    # Create a Pool of worker processes to handle the chunks in parallel
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        result = pool.map(process_chunk, chunks)  # Process the chunks in parallel

    # Flatten the list of results if necessary
    return [item for sublist in result for item in sublist]

def run_experiment_a_old(parameter_combinations, file_path="results_vect1.csv"):
    """Run the experiment for all parameter combinations using multiprocessing."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() // 3) as executor:
        futures = [
            executor.submit(run_experiment_chunk_vect, [comb])  # Submit tasks for each combination
            for comb in parameter_combinations
        ]
        
        for completed, future in enumerate(concurrent.futures.as_completed(futures)):
            results_chunk = future.result()  # Get the result from the future
            save_results(results_chunk, file_path)
            progress = (completed + 1) / len(parameter_combinations) * 100
            print(f"\rProgress: {progress:.2f}% ({completed + 1}/{len(parameter_combinations)})", end="")

    print("\nProcessing complete.")



def run_experiment_with_timeout(comb, time_limit):
    """Attempt to run a combination with a retry mechanism."""
    retries = 0  # Track the number of retries
    
    # Start the timer for the experiment
    start_time = time.time()

    while retries < 2:  # Retry only once (for a total of 2 attempts)
        try:
            # Check the elapsed time before each run
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                raise TimeoutError(f"Timeout occurred for combination {comb} after {elapsed_time:.2f} seconds.")
            
            # Run the experiment
            results_chunk = run_experiment_chunk_vect([comb])
            
            # Return results if successful
            return results_chunk  
        except TimeoutError as e:
            print(f"Error with combination {comb}: {e}")
            retries += 1  # Increment retry counter
            if retries < 2:
                print(f"Retrying {comb} (attempt #{retries})...")
                start_time = time.time()  # Restart the timer for the next attempt
        except Exception as e:
            print(f"Unexpected error with combination {comb}: {e}")
            break  # Break out of the loop for any unexpected error

    print(f"Skipping combination {comb} after {retries} retries.")
    return None  # Return None if both attempts failed

def run_experiment_a(parameter_combinations, file_path="results_vect_lambda.csv", time_limit=300):
    """Run the experiment for all parameter combinations using multiprocessing with a time limit for each task."""
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() // 3) as executor:
        futures = [
            executor.submit(run_experiment_with_timeout, comb, time_limit)  # Submit tasks for each combination
            for comb in parameter_combinations
        ]
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                results_chunk = future.result(timeout=time_limit)  # Ensure timeout for each result
                if results_chunk is not None:
                    save_results(results_chunk, file_path)  # Save the results if no timeout occurs
                else:
                    print(f"Skipping combination due to timeout or error.")
                
                completed += 1  # Track progress
                progress = (completed) / len(parameter_combinations) * 100
                print(f"\rProgress: {progress:.2f}% ({completed}/{len(parameter_combinations)})", end="")
            except concurrent.futures.TimeoutError:
                print(f"Timeout occurred for a parameter combination, retrying.")
            except Exception as e:
                print(f"Unexpected error: {e}")

    print("\nProcessing complete.")


# Test the parallel combination generation
if __name__ == "__main__":
    alpha_values = np.linspace(0.01, 0.5, 3)
    lambda_values = np.linspace(0.01, 0.99, 25)
    delta_values = np.linspace(0.01, 0.75, 3)
    epsilon_values = np.linspace(0.01, 0.5, 2)
    M = np.linspace(100, 10000, 5)


    # Function to round each linspace value to the nearest multiple of 3
    def round_to_nearest_multiple(values, multiple):
        return np.round(values * multiple) / multiple

    # Round all the values to the nearest multiple of 3
    alpha_values_rounded = round_to_nearest_multiple(alpha_values, 3)
    lambda_values_rounded = round_to_nearest_multiple(lambda_values, 3)
    delta_values_rounded = round_to_nearest_multiple(delta_values, 3)
    epsilon_values_rounded = round_to_nearest_multiple(epsilon_values, 3)
    M_rounded = np.round(M)

    random_uniform_values = [False]

    # Generate parameter combinations in parallel
    parameter_combinations = generate_combinations_parallel(alpha_values, lambda_values, epsilon_values, delta_values, random_uniform_values, M_rounded)
    
    # Run the experiment with the generated combinations
    print(f"Generated {len(parameter_combinations)} combinations.")
    run_experiment_a(parameter_combinations)

