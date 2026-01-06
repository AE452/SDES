# Asynchronous Decentralized Q-Learning for Multi-Agent Games

This repository contains the implementation and analysis tools for studying asynchronous decentralized Q-learning in multi-agent game theory settings, with a focus on the Prisoner's Dilemma game.

## Overview

This project investigates the dynamics of Q-learning algorithms in multi-agent environments, specifically analyzing:
- **Basin of Attraction (BoA)**: Geometric analysis of convergence regions for different strategy equilibria
- **Strategy Evolution**: How agents learn and converge to different strategies (Always Defect, Grim Trigger, etc.)
- **Equilibrium Analysis**: Theoretical and empirical analysis of game equilibria in both single-agent and multi-agent settings

## Key Features

- **Multi-Agent Q-Learning**: Implementation of asynchronous decentralized Q-learning for two-player games
- **Basin of Attraction Visualization**: Tools for plotting and analyzing convergence regions
- **Strategy Detection**: Automatic detection of learned strategies from Q-value trajectories
- **Theoretical Q-Value Calculations**: Analytical solutions for Q-values under different strategies
- **Vectorized Simulations**: Efficient batch processing of multiple parameter combinations

## Project Structure

```
.
├── MultiAgent/              # Multi-agent Q-learning implementations
│   ├── MultiAgent_DQ.py     # Main multi-agent Q-learning algorithm
│   ├── basin.py             # Basin of attraction analysis and visualization
│   ├── sim_script_vectorized.py  # Vectorized simulation script
│   └── comparison_algorithms.py  # Algorithm comparison utilities
├── SingleAgent/             # Single-agent analysis
│   └── SingleAgent.py       # Single-agent Q-value calculations
├── Pricing/                 # Pricing model implementations
│   └── dynamicdemand.py     # Dynamic pricing and demand models
├── examples/                # Example scripts and usage demonstrations
├── utils.py                 # Utility functions for data processing and visualization
└── dashboard.py             # Interactive dashboard for results visualization
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Async_Dec_Q
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Multi-Agent Q-Learning

```python
from MultiAgent.MultiAgent_DQ import Algorithm

# Initialize algorithm with parameters
algo = Algorithm(
    e=0.1,              # Exploration parameter (epsilon)
    lambda_val=0.5,     # Eligibility trace parameter
    alpha=0.1,          # Learning rate
    delta=0.7,          # Discount factor
    init_mode="random", # Initialization mode
    tolerance=1e-5,     # Convergence tolerance
    M=500,              # Number of samples
    verbose=True
)

# Run the algorithm
q1_history, q2_history, batch_times = algo.run()
```

### Basin of Attraction Analysis

```python
from MultiAgent.basin import VectorFieldAutomatic
from MultiAgent.basin import QvaluesADvsAD, QvaluesGTvsGT

# Create vector field plotter
plotter = VectorFieldAutomatic(
    func1=QvaluesADvsAD,
    func2=QvaluesGTvsGT,
    strategy1="AD",
    strategy2="GT",
    epsilon=0.1,
    d=0.7,
    show_basin=True,
    show_trajectory=True
)

# Plot the basin of attraction
plotter.plot_basin()
```

### Running Simulations

For batch simulations with multiple parameter combinations, see `examples/run_simulation.py`.

## Game Parameters

The default Prisoner's Dilemma payoff matrix is:

| Player A / Player B | Cooperate (C) | Defect (D) |
|---------------------|---------------|------------|
| **Cooperate (C)**   | R = 0.8       | S = 0      |
| **Defect (D)**      | T = 1.0       | P = 0.2    |

### Q-Value Matrix Notation

The Q-value matrix is structured as:
```
Q = [[Q_00, Q_01],
     [Q_10, Q_11]]
```

Where:
- First subscript (0 or 1) represents the action: 0 = Cooperate, 1 = Defect
- Second subscript (0 or 1) represents the state: 0 = State C (CC), 1 = State D (CD, DC, DD)

## Strategies

The code supports analysis of four main strategies:

- **AD (Always Defect)**: Agent always defects regardless of opponent's action
- **GT (Grim Trigger)**: Agent cooperates until opponent defects, then defects forever
- **AGT (Anti-Grim Trigger)**: Variant of Grim Trigger
- **AC (Always Cooperate)**: Agent always cooperates

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your-paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal of Machine Learning Research},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on the repository or contact the authors.

## Acknowledgments

This work builds upon research in multi-agent reinforcement learning and game theory.
