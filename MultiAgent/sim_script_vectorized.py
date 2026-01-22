import itertools
import time
from MultiAgent_DQ import Algorithm_1 as Algorithm
from utils import convert_dict_to_mean_df
import pandas as pd
import numpy as np
import os
import gc
import concurrent.futures
import multiprocessing
from gc import collect
import logging
from datetime import datetime

#this has been the function used for the simulations
def run_experiment_chunk_vect(parameter_combinations_chunk):
    
    results_chunk = []

    for idx, (M_value, alpha, lambda_val, e, delta, random_uniform) in enumerate(parameter_combinations_chunk):
        algo = Algorithm(
            e=e, lambda_val=lambda_val, alpha=alpha, delta=delta,
            random_uniform=random_uniform, tolerance=1e-5, 
            M=int(M_value), verbose=False
        )
        q1_history_per_sample, q2_history_per_sample, batch_times, timeout_out_flag_dict, timeout_vi_flag_dict = algo.run()

        # Process results
        dfs_1 = convert_dict_to_mean_df(q1_history_per_sample, mean=False)
        dfs_2 = convert_dict_to_mean_df(q2_history_per_sample, mean=False)
        batch_times_df = pd.DataFrame(list(batch_times.items()), columns=["Batch", "Time"])
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

        batch_times_df_max = batch_times_df['Time'].max()
        batch_times_df_min = batch_times_df['Time'].min()
        batch_times_df_mean = batch_times_df['Time'].mean()
        # Save result
        results_chunk.append({'M': int(M_value),
            'alpha': alpha, 'lambda': lambda_val, 'e': e, 'delta': delta,
            'random_uniform': random_uniform, 'P_AD': P_AD, 'P_GT': P_GT,
            'nonconv': nonconv, 'time_max': batch_times_df_max, 'time_min': batch_times_df_min, 'time_mean': batch_times_df_mean
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

    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
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

def generate_combinations_parallel(M, alpha_values, lambda_values, epsilon_values, delta_values, random_uniform_values):
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

def run_experiment_a_old(parameter_combinations, file_path="results_vect_fixed.csv"):
    """Run the experiment for all parameter combinations using multiprocessing."""
    with concurrent.futures.ProcessPoolExecutor(max_workers= 3) as executor:
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

def run_experiment_with_timeout(comb, time_limit, comb_index=None):
    """Attempt to run a combination with a retry mechanism."""
    retries = 0  # Track the number of retries
    
    # Start the timer for the experiment
    start_time = time.time()
    
    comb_str = f"Index {comb_index}: {comb}" if comb_index is not None else str(comb)

    while retries < 1:  # Retry only once (for a total of 2 attempts)
        try:
            # Check the elapsed time before each run
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                error_msg = f"Timeout occurred for combination {comb_str} after {elapsed_time:.2f} seconds."
                logging.error(error_msg)
                raise TimeoutError(error_msg)
            
            # Run the experiment
            logging.info(f"Starting combination {comb_str}")
            results_chunk = run_experiment_chunk_vect([comb])
            logging.info(f"Successfully completed combination {comb_str}")
            
            # Return results if successful
            return results_chunk  
        except TimeoutError as e:
            error_msg = f"Timeout error with combination {comb_str}: {e}"
            logging.error(error_msg)
            print(f"\n{error_msg}")
            retries += 1  # Increment retry counter
            if retries < 2:
                print(f"Retrying {comb_str} (attempt #{retries})...")
                start_time = time.time()  # Restart the timer for the next attempt
        except Exception as e:
            error_msg = f"Unexpected error with combination {comb_str}: {e}"
            logging.error(error_msg, exc_info=True)
            print(f"\n{error_msg}")
            import traceback
            traceback.print_exc()
            break  # Break out of the loop for any unexpected error

    print(f"Skipping combination {comb_str} after {retries} retries.")
    return None  # Return None if both attempts failed

def run_experiment_a(parameter_combinations, file_path="results_vect_paramsfixed_time.csv", time_limit=300, start_index=0):
    """Run the experiment for all parameter combinations using multiprocessing with a time limit for each task.
    
    Parameters:
    -----------
    parameter_combinations : list
        List of parameter combinations to process
    file_path : str
        Path to save results CSV file
    time_limit : int
        Time limit in seconds for each combination
    start_index : int
        Starting index for progress tracking (useful when resuming from a specific point)
    """
    
    # Setup logging
    log_file = file_path.replace('.csv', '_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting experiment with {len(parameter_combinations)} combinations, starting from index {start_index}")
    logging.info(f"Results will be saved to: {file_path}")
    logging.info(f"Log file: {log_file}")
    
    total_combinations = len(parameter_combinations) + start_index
    
    # Save checkpoint file with current progress
    checkpoint_file = file_path.replace('.csv', '_checkpoint.txt')
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers= 3) as executor:
            # Create a mapping of futures to their indices and combinations for better error tracking
            future_to_info = {}
            for idx, comb in enumerate(parameter_combinations):
                current_index = start_index + idx
                future = executor.submit(run_experiment_with_timeout, comb, time_limit, current_index)
                future_to_info[future] = {
                    'index': current_index,
                    'combination': comb
                }
            
            completed = 0
            failed_combinations = []
            
            for future in concurrent.futures.as_completed(future_to_info):
                info = future_to_info[future]
                current_index = info['index']
                comb = info['combination']
                
                # Update checkpoint
                with open(checkpoint_file, 'w') as f:
                    f.write(f"Last processed index: {current_index}\n")
                    f.write(f"Completed: {completed}/{len(parameter_combinations)}\n")
                    f.write(f"Combination: {comb}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                
                try:
                    results_chunk = future.result(timeout=time_limit + 10)  # Add small buffer to timeout
                    if results_chunk is not None:
                        save_results(results_chunk, file_path)  # Save the results if no timeout occurs
                        logging.info(f"Successfully saved results for index {current_index}")
                    else:
                        error_msg = f"Skipping combination at index {current_index} due to timeout or error. Combination: {comb}"
                        logging.warning(error_msg)
                        print(f"\n{error_msg}")
                        failed_combinations.append((current_index, comb, "Timeout or error"))
                    
                    completed += 1  # Track progress
                    current_absolute = start_index + completed
                    progress = (completed) / len(parameter_combinations) * 100
                    print(f"\rProgress: {progress:.2f}% ({current_absolute}/{total_combinations}) - Completed: {completed}/{len(parameter_combinations)}", end="")
                    
                except concurrent.futures.TimeoutError:
                    error_msg = f"Timeout occurred for combination at index {current_index}. Combination: {comb}"
                    logging.error(error_msg)
                    print(f"\n{error_msg}")
                    failed_combinations.append((current_index, comb, "Timeout"))
                    
                except Exception as e:
                    error_msg = f"Unexpected error at index {current_index}: {e}. Combination: {comb}"
                    logging.error(error_msg, exc_info=True)
                    print(f"\n{error_msg}")
                    import traceback
                    traceback.print_exc()
                    failed_combinations.append((current_index, comb, str(e)))
                    
                    # Save error details to a separate file
                    error_file = file_path.replace('.csv', '_errors.txt')
                    with open(error_file, 'a') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"Index: {current_index}\n")
                        f.write(f"Combination: {comb}\n")
                        f.write(f"Error: {e}\n")
                        f.write(f"Traceback:\n")
                        traceback.print_exc(file=f)
                        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        
        # Summary
        logging.info(f"Processing complete. Total: {completed}/{len(parameter_combinations)}")
        if failed_combinations:
            logging.warning(f"Failed combinations: {len(failed_combinations)}")
            for idx, comb, error in failed_combinations:
                logging.warning(f"  Index {idx}: {error} - {comb}")
        
        print("\nProcessing complete.")
        if failed_combinations:
            print(f"\nWarning: {len(failed_combinations)} combinations failed. Check log file: {log_file}")
            
    except KeyboardInterrupt:
        logging.warning("Experiment interrupted by user")
        print("\n\nExperiment interrupted. Checkpoint saved.")
        raise
    except Exception as e:
        logging.critical(f"Critical error in run_experiment_a: {e}", exc_info=True)
        print(f"\n\nCritical error: {e}")
        print(f"Check log file for details: {log_file}")
        raise


def read_checkpoint(file_path="results_vect_paramsfixed_time.csv"):
    """Read the last checkpoint to see where the experiment stopped."""
    checkpoint_file = file_path.replace('.csv', '_checkpoint.txt')
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            content = f.read()
            print("Last checkpoint:")
            print(content)
            return content
    else:
        print("No checkpoint file found.")
        return None


# Test the parallel combination generation
if __name__ == "__main__":
    alpha_values = [0.01, 0.051, 0.092, 0.134, 0.175, 0.216, 0.258, 0.299, 0.34, 0.381, 0.422, 0.464, 0.505, 0.546, 0.588, 0.629, 0.67, 0.711, 0.753, 0.794, 0.835, 0.876, 0.918, 0.959,1]
    lambda_values = [0.01, 0.051, 0.092, 0.134, 0.175, 0.216, 0.258, 0.299, 0.34, 0.381, 0.422, 0.464, 0.505, 0.546, 0.588, 0.629, 0.67, 0.711, 0.753, 0.794, 0.835, 0.876, 0.918, 0.959]
    M_values = [20, 100, 500]
    epsilon_values = [0.05, 0.35]
    delta_values = [0.5, 0.75, 0.95] 


    # Function to round each linspace value to the nearest multiple of 3
    def round_to_nearest_multiple(values, multiple):
        return np.round(values * multiple) / multiple

    # Round all the values to the nearest multiple of 3
    alpha_values_rounded = round_to_nearest_multiple(alpha_values, 3)
    lambda_values_rounded = round_to_nearest_multiple(lambda_values, 3)
    delta_values_rounded = round_to_nearest_multiple(delta_values, 3)
    epsilon_values_rounded = round_to_nearest_multiple(epsilon_values, 3)
    M_rounded = np.round(M_values)

    random_uniform_values = [False]

    # Generate parameter combinations in parallel
    parameter_combinations = generate_combinations_parallel(M_rounded, alpha_values, lambda_values, epsilon_values, delta_values, random_uniform_values)
    
    # Run the experiment with the generated combinations
    print(f"Generated {len(parameter_combinations)} combinations.")
    start_idx = 7339
    run_experiment_a(parameter_combinations[start_idx:], start_index=start_idx)

