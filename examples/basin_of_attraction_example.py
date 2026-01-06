"""
Example: Basin of Attraction Visualization

This example demonstrates how to:
1. Create a basin of attraction plot for different strategy pairs
2. Visualize the convergence regions for game equilibria
3. Analyze the dynamics of Q-value trajectories
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MultiAgent.basin import VectorFieldAutomatic
from MultiAgent.basin import QvaluesADvsAD, QvaluesGTvsGT, QvaluesADvsAC, QvaluesADvsAGT
import numpy as np

def example_ad_vs_gt():
    """
    Example: Always Defect vs Grim Trigger
    """
    print("=" * 60)
    print("Basin of Attraction: AD vs GT")
    print("=" * 60)
    
    # Game parameters
    epsilon = 0.1
    delta = 0.7
    t, r, p, s = 1.0, 0.8, 0.2, 0.0
    
    # Create vector field plotter
    plotter = VectorFieldAutomatic(
        func1=QvaluesADvsAD,
        func2=QvaluesGTvsGT,
        strategy1="AD",
        strategy2="GT",
        t=t, r=r, p=p, s=s,
        epsilon=epsilon,
        d=delta,
        show_basin=True,
        show_trajectory=False,  # Set to True if you have Q-value trajectories
        alpha=0.3,
        density=15
    )
    
    # Plot the basin of attraction
    print("Generating basin of attraction plot...")
    plotter.plot_basin()
    print("Plot generated successfully!")

def example_ad_vs_ac():
    """
    Example: Always Defect vs Always Cooperate
    """
    print("\n" + "=" * 60)
    print("Basin of Attraction: AD vs AC")
    print("=" * 60)
    
    # Game parameters
    epsilon = 0.1
    delta = 0.7
    t, r, p, s = 1.0, 0.8, 0.2, 0.0
    
    # Create vector field plotter
    plotter = VectorFieldAutomatic(
        func1=QvaluesADvsAD,
        func2=QvaluesADvsAC,
        strategy1="AD",
        strategy2="AC",
        t=t, r=r, p=p, s=s,
        epsilon=epsilon,
        d=delta,
        show_basin=True,
        show_trajectory=False,
        alpha=0.3,
        density=15
    )
    
    # Plot the basin of attraction
    print("Generating basin of attraction plot...")
    plotter.plot_basin()
    print("Plot generated successfully!")

def main():
    """
    Run basin of attraction examples.
    """
    print("Basin of Attraction Visualization Examples")
    print("=" * 60)
    
    # Example 1: AD vs GT
    example_ad_vs_gt()
    
    # Example 2: AD vs AC (uncomment to run)
    # example_ad_vs_ac()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNote: In Jupyter notebooks, plots are displayed automatically.")
    print("In scripts, you may need to call plt.show() or save figures explicitly.")

if __name__ == "__main__":
    main()

