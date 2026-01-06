"""
Basic example: Running multi-agent Q-learning algorithm

This example demonstrates how to:
1. Initialize the multi-agent Q-learning algorithm
2. Run the algorithm with specified parameters
3. Extract and examine the results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MultiAgent.MultiAgent_DQ import Algorithm
import numpy as np

def main():
    """
    Run a basic multi-agent Q-learning simulation.
    """
    print("=" * 60)
    print("Multi-Agent Q-Learning Example")
    print("=" * 60)
    
    # Algorithm parameters
    e = 0.1              # Exploration parameter (epsilon)
    lambda_val = 0.5     # Inertia parameter
    alpha = 0.1          # Learning rate
    delta = 0.7          # Discount factor
    init_mode = "latin"  # Initialization mode: "latin", "uniform", or "fixed"
    tolerance = 1e-5     # Convergence tolerance
    M = 10               # Number of samples (use small number for quick demo)
    verbose = True       # Print progress
    
    print(f"\nParameters:")
    print(f"  Epsilon (e): {e}")
    print(f"  Inertia: {lambda_val}")
    print(f"  Alpha: {alpha}")
    print(f"  Delta: {delta}")
    print(f"  Init mode: {init_mode}")
    print(f"  Tolerance: {tolerance}")
    print(f"  Number of samples (M): {M}")
    print()
    
    # Initialize algorithm
    algo = Algorithm(
        e=e,
        lambda_val=lambda_val,
        alpha=alpha,
        delta=delta,
        init_mode=init_mode,
        tolerance=tolerance,
        M=M,
        verbose=verbose
    )
    
    # Run the algorithm
    print("Running algorithm...")
    q1_history_per_sample, q2_history_per_sample, batch_times = algo.run()
    
    # Display results summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Number of successful runs: {len(q1_history_per_sample)}")
    print(f"Number of batches: {len(batch_times)}")
    
    # Show final Q-values for first sample
    if len(q1_history_per_sample) > 0:
        sample_key = list(q1_history_per_sample.keys())[0]
        q1_final = q1_history_per_sample[sample_key][-1]
        q2_final = q2_history_per_sample[sample_key][-1]
        
        print(f"\nFinal Q-values for sample {sample_key}:")
        print(f"  Agent 1 Q-matrix:")
        print(f"    {q1_final}")
        print(f"  Agent 2 Q-matrix:")
        print(f"    {q2_final}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()

