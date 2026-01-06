"""
Example: Comparing Different Strategy Initializations

This example demonstrates how to:
1. Run the algorithm with fixed strategy initializations
2. Compare convergence behavior for different strategy pairs
3. Analyze Q-value evolution for specific strategy combinations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MultiAgent.MultiAgent_DQ import Algorithm
import numpy as np

def run_strategy_pair(strategy1, strategy2, M=5):
    """
    Run the algorithm with a specific strategy pair initialization.
    
    Parameters
    ----------
    strategy1 : str
        Strategy for agent 1: "AD", "GT", "AGT", or "AC"
    strategy2 : str
        Strategy for agent 2: "AD", "GT", "AGT", or "AC"
    M : int
        Number of samples to run
    
    Returns
    -------
    dict
        Results dictionary with Q-value histories
    """
    print(f"\nRunning: {strategy1} vs {strategy2}")
    print("-" * 40)
    
    # Algorithm parameters
    e = 0.1
    lambda_val = 0.5
    alpha = 0.1
    delta = 0.7
    tolerance = 1e-5
    
    # Initialize algorithm with fixed strategies
    algo = Algorithm(
        e=e,
        lambda_val=lambda_val,
        alpha=alpha,
        delta=delta,
        init_mode="fixed",
        tolerance=tolerance,
        M=M,
        strategy1=strategy1,
        strategy2=strategy2,
        verbose=False
    )
    
    # Run the algorithm
    q1_history, q2_history, batch_times = algo.run()
    
    # Calculate convergence statistics
    num_converged = len(q1_history)
    avg_batches = np.mean(list(batch_times.values())) if batch_times else 0
    
    print(f"  Converged runs: {num_converged}/{M}")
    print(f"  Average batches: {avg_batches:.2f}")
    
    return {
        'strategy1': strategy1,
        'strategy2': strategy2,
        'q1_history': q1_history,
        'q2_history': q2_history,
        'batch_times': batch_times,
        'num_converged': num_converged
    }

def main():
    """
    Compare different strategy pairs.
    """
    print("=" * 60)
    print("Strategy Comparison Example")
    print("=" * 60)
    
    # Strategy pairs to compare
    strategy_pairs = [
        ("AD", "AD"),
        ("GT", "GT"),
        ("AD", "GT"),
        ("GT", "AD"),
    ]
    
    results = {}
    
    for strategy1, strategy2 in strategy_pairs:
        result = run_strategy_pair(strategy1, strategy2, M=5)
        key = f"{strategy1}-{strategy2}"
        results[key] = result
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for key, result in results.items():
        print(f"{key:15s}: {result['num_converged']}/5 converged")
    
    print("\nExample completed!")

if __name__ == "__main__":
    main()

