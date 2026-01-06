# Examples

This directory contains example scripts demonstrating how to use the codebase.

## Available Examples

### 1. `basic_multiagent_example.py`
Basic example showing how to:
- Initialize the multi-agent Q-learning algorithm
- Run simulations with different parameter settings
- Extract and examine results

**Usage:**
```bash
python examples/basic_multiagent_example.py
```

### 2. `basin_of_attraction_example.py`
Demonstrates basin of attraction visualization:
- Create plots for different strategy pairs
- Visualize convergence regions
- Analyze Q-value dynamics

**Usage:**
```bash
python examples/basin_of_attraction_example.py
```

**Note:** In Jupyter notebooks, plots are displayed automatically. In scripts, you may need to add `plt.show()` or save figures explicitly.

### 3. `strategy_comparison_example.py`
Compares different strategy initializations:
- Run algorithm with fixed strategy pairs
- Compare convergence behavior
- Analyze Q-value evolution

**Usage:**
```bash
python examples/strategy_comparison_example.py
```

## Running Examples

All examples assume you are running from the project root directory. If running from a different location, adjust the `sys.path.append()` statements in each script accordingly.

## Customization

Feel free to modify the parameters in these examples to explore different:
- Exploration rates (epsilon)
- Learning rates (alpha)
- Discount factors (delta)
- Strategy combinations
- Number of samples

