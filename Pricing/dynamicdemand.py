import numpy as np
from itertools import product
from scipy.optimize import fsolve
import sys
import copy
"""
Dynamic Pricing and Demand Evolution Model
How It Works:
Initialization:

Prices start at a default value.
Demand and profits are calculated for the initial state.
Dynamic Updates:

Demand: A weighted average of past demand and current demand based on prices.
Prices: Adjusted using a gradient-based rule, ensuring firms learn from demand/profit signals.
Profits: Computed at each step using the updated prices and demand.
Simulation Loop:

The process is repeated for a fixed number of time steps (tmax).


Results:

At the end of the simulation, the final prices, demand, and profits reflect a steady state (if equilibrium is reached).
"""

class IRP(object):
    """
    IRP Model

    Attributes
    ----------
    n : int
        Number of players.
    alpha : float
        Product differentiation parameter.
    mu : float
        Product differentiation parameter.
    a : int
        Value of the products.
    a0 : float
        Value of the outside option.
    c : float
        Marginal cost.
    tstable : int
        Periods of game stability.
    tmax : int
        Maximum iterations of play.
    dem_function : str or callable
        Demand function to use ('default' or custom function).
    """

    def __init__(self, dem_function='default', **kwargs):
        """
        Initialize the IRP model with given or default parameters.

        Parameters
        ----------
        dem_function : str or callable, optional
            Demand function to use (default is 'default').
        **kwargs : dict
            Additional parameters to override default values.
        """
        # Default properties
        self.n = kwargs.get('n', 2)
        self.alpha = kwargs.get('alpha', 0.15)
        self.c = kwargs.get('c', 1)
        self.a = kwargs.get('a', 2)
        self.a0 = kwargs.get('a0', 0)
        self.mu = kwargs.get('mu', 0.25)
        self.tstable = kwargs.get('tstable', 1e2)
        self.tmax = kwargs.get('tmax', 1e4)

        self.dem_function = dem_function

        # Derived properties
        # self.sdim, self.s0 = self.init_state()
        # self.p_minmax = self.compute_p_competitive_monopoly()
        # self.A = self.init_actions()
        # self.PI = self.init_PI()

    def demand(self, p):
        """
        Compute the demand for each firm given their prices.

        Parameters
        ----------
        p : ndarray
            Array of prices set by each firm.

        Returns
        -------
        ndarray
            Array of demand quantities for each firm.
        """
        e = np.exp((self.a - p) / self.mu)
        d = e / (np.sum(e) + np.exp(self.a0 / self.mu))
        return d

    def foc(self, p):
        """
        Compute the first-order condition for profit maximization.

        Parameters
        ----------
        p : ndarray
            Array of prices set by each firm.

        Returns
        -------
        ndarray
            Array of first-order condition values.
        """
        if self.dem_function == 'default':
            d = self.demand(p)
        else:
            d = self.dem_function(p)
        zero = 1 - (p - self.c) * (1 - d) / self.mu
        return np.squeeze(zero)

    def foc_monopoly(self, p):
        """
        Compute the first-order condition for a monopolist.

        Parameters
        ----------
        p : ndarray
            Array of prices set by the monopolist for each product.

        Returns
        -------
        ndarray
            Array of first-order condition values for a monopolist.
        """
        if self.dem_function == 'default':
            d = self.demand(p)
        else:
            d = self.dem_function(p)
        d1 = np.flip(d)
        p1 = np.flip(p)
        zero = 1 - (p - self.c) * (1 - d) / self.mu + (p1 - self.c) * d1 / self.mu
        return np.squeeze(zero)

    def compute_p_competitive_monopoly(self):
        """
        Compute competitive and monopoly prices.

        Returns
        -------
        tuple
            A tuple containing competitive and monopoly prices.
        """
        p0 = np.ones((1, self.n)) * 3 * self.c
        p_competitive = fsolve(self.foc, p0)
        p_monopoly = fsolve(self.foc_monopoly, p0)
        return p_competitive, p_monopoly

    def init_actions(self):
        """
        Initialize the discrete action space (possible prices).

        Returns
        -------
        ndarray
            Array of possible prices.
        """
        a = np.linspace(min(self.p_minmax[0]), max(self.p_minmax[1]), self.k - 2)
        delta = a[1] - a[0]
        A = np.linspace(min(a) - delta, max(a) + delta, self.k)
        return A

    def init_state(self):
        """
        Initialize the state space dimensions and initial state.

        Returns
        -------
        tuple
            A tuple containing state space dimensions and initial state.
        """
        sdim = (self.k, self.k)
        s0 = np.zeros(len(sdim)).astype(int)
        return sdim, s0

    def compute_profits(self, p):
        """
        Compute profits for each firm given their prices.

        Parameters
        ----------
        p : ndarray
            Array of prices set by each firm.

        Returns
        -------
        ndarray
            Array of profits for each firm.
        """ 
        d = self.demand(p)
        pi = (p - self.c) * d
        return pi

    def init_PI(self, game):
        """
        Initialize the profit matrix for all possible states and actions.

        Parameters
        ----------
        game : IRP
            The game environment.

        Returns
        -------
        ndarray
            3D array of profits for all possible states and actions.
        """
        PI = np.zeros(game.sdim + (game.n,))
        for s in product(*[range(i) for i in game.sdim]):
            p = np.asarray(game.A[np.asarray(s)])
            PI[s] = game.compute_profits(p)
        return PI

    def show_stats(self):
        """
        Display competitive and monopoly prices.
        """
        p_competitive, p_monopoly = self.compute_p_competitive_monopoly()
        return  p_competitive, p_monopoly

class IRP_Dynamic(IRP):
    def __init__(self, dem_function='default', **kwargs):
        super().__init__(dem_function, **kwargs)
        self.learning_rate = kwargs.get('learning_rate', 0.1)  # Price adjustment step size
        self.memory_factor = kwargs.get('memory_factor', 0.5)  # Demand memory factor
        self.tmax = kwargs.get('tmax', 100)  # Number of time steps

    def dynamic_simulation(self):
        """
        Simulate the dynamic pricing and demand evolution.
        """
        # Initialize prices, demands, and profits
        prices = np.ones((self.tmax, self.n)) * (3 * self.c)
        demands = np.zeros((self.tmax, self.n))
        profits = np.zeros((self.tmax, self.n))

        # Initial demand and profit
        demands[0] = self.demand(prices[0])
        profits[0] = self.compute_profits(prices[0])

        # Iterate over time steps
        for t in range(1, self.tmax):
            # Compute new demand based on lagged adjustment
            demands[t] = (
                self.memory_factor * demands[t - 1] +
                (1 - self.memory_factor) * self.demand(prices[t - 1])
            )

            # Update prices based on profit gradient
            profit_grad = (demands[t] - prices[t - 1]) / self.mu
            prices[t] = prices[t - 1] + self.learning_rate * profit_grad

            # Ensure prices are non-negative
            prices[t] = np.maximum(prices[t], 0)

            # Compute profits
            profits[t] = self.compute_profits(prices[t])

        return prices, demands, profits

    def show_dynamic_results(self):
        """
        Run the simulation and print the results.
        """
        prices, demands, profits = self.dynamic_simulation()
        print("Final Prices:", prices[-1])
        print("Final Demands:", demands[-1])
        print("Final Profits:", profits[-1])
        return prices, demands, profits
