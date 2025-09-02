import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# --------------------------------------------------------
# Geometric interpretation of strategies and Best Responses
# 
#Looking at the IBRG, we can see just when there are two equilibria, we are interested in analyzing the BoA and the dynamics
# 
# --------------------------------------------------------
# Best Response (BR) mapping for discrete strategies:
#
# Each strategy is a discrete action: AD, GT, AGT, AC.
# The BR function maps the opponent's strategy to the optimal response:
#
#   Opponent  -> Best Response
#   --------------------------
#   AD        -> AD
#   GT        -> GT
#   AGT       -> AD
#   AC        -> AD
#
# Interpretation:
# - AD and GT are self-consistent: if the opponent plays AD or GT, the BR is the same → equilibrium.
# - AGT and AC are unstable: if the opponent plays AGT or AC, the BR is AD
#
# --------------------------------------------------------
# Summary:
#   - AD and GT are self-consistent stable strategies (pure-strategy equilibria).
#   - AGT and AC are unstable: the BR points "up" toward AD.
#   - This illustrates a geometric dynamic where the upper part of the plane
#     is stable and the lower part "flows" toward the top-right (AD).
# --------------------------------------------------------


def QvaluesADvsAD(t, r, p, s, epsilon, d):
    denom = 2 * (d - 1)  # equivalent to 2*(-1 + d)
    
    first_term = -((2*p - epsilon*p + epsilon*t) / denom)
    second_term = -( (2*d*p - d*epsilon*p + epsilon*r - d*epsilon*r 
                      + 2*s - 2*d*s - epsilon*s + d*epsilon*s + d*epsilon*t) / denom )
    
    return np.array([[first_term, second_term],
                     [first_term, second_term]])


def QvaluesADvsAGT(t, r, p, s, epsilon, d):
    denom = 2 * (d - 1)  # same as 2*(-1 + d)

    first_term_1 = -((2*p - 2*d*p - epsilon*p + 2*d*epsilon*p + 2*d*t + epsilon*t - 2*d*epsilon*t) / denom)
    second_term_1 = -((2*d*epsilon*p - d**2*epsilon*p - d*epsilon**2*p + d**2*epsilon**2*p
                       + epsilon*r - d*epsilon*r + 2*s - 2*d*s - epsilon*s + d*epsilon*s
                       + 2*d*t - 2*d*epsilon*t + d**2*epsilon*t + d*epsilon**2*t - d**2*epsilon**2*t) / denom)

    first_term_2 = -((epsilon*p + 2*t - epsilon*t) / denom)
    second_term_2 = -((2*d*p - 2*d**2*p - 2*d*epsilon*p + 3*d**2*epsilon*p
                       + d*epsilon**2*p - d**2*epsilon**2*p
                       + 2*r - 2*d*r - epsilon*r + d*epsilon*r
                       + epsilon*s - d*epsilon*s
                       + 2*d**2*t + 2*d*epsilon*t - 3*d**2*epsilon*t
                       - d*epsilon**2*t + d**2*epsilon**2*t) / denom)

    return np.array([[first_term_1, second_term_1],
                     [first_term_2, second_term_2]])


def QvaluesGTvsGT(t, r, p, s, epsilon, d):
    denom1 = 2 * (d - 1)  # same as 2*(-1 + d)
    denom2 = (2 - 2*d + d*epsilon)  # extra denominator factor

    first_term_1 = -((2*d*p + epsilon*p - 2*d*epsilon*p
                      + 2*t - 2*d*t - epsilon*t + 2*d*epsilon*t) / denom1)

    second_term_1 = -((2*d*epsilon*p - d*epsilon**2*p
                       + 4*r - 4*d*r - 2*epsilon*r + 2*d*epsilon*r
                       + 2*epsilon*s - 2*d*epsilon*s
                       + d*epsilon**2*t) / (denom1 * denom2))

    first_term_2 = -((2*p - epsilon*p + epsilon*t) / denom1)

    second_term_2 = -((4*d*p - 4*d**2*p - 4*d*epsilon*p + 6*d**2*epsilon*p
                       + d*epsilon**2*p - 2*d**2*epsilon**2*p
                       + 2*epsilon*r - 2*d*epsilon*r
                       + 4*s - 8*d*s + 4*d**2*s
                       - 2*epsilon*s + 6*d*epsilon*s - 4*d**2*epsilon*s
                       + 2*d*epsilon*t - 2*d**2*epsilon*t
                       - d*epsilon**2*t + 2*d**2*epsilon**2*t) / (denom1 * denom2))

    return np.array([[first_term_1, second_term_1],
                     [first_term_2, second_term_2]])


def QvaluesADvsAC(t, r, p, s, epsilon, d):
    denom = 2 * (d - 1)  # same as 2*(-1 + d)

    first_term = -((epsilon*p + 2*t - epsilon*t) / denom)
    second_term = -((d*epsilon*p + 2*r - 2*d*r - epsilon*r + d*epsilon*r
                     + epsilon*s - d*epsilon*s + 2*d*t - d*epsilon*t) / denom)

    return np.array([[first_term, second_term],
                     [first_term, second_term]])


class VectorFieldPlot:
    """
    Plots vector fields from two Q-functions (func1 for x, func2 for y),
    with trajectories from Q-learning runs and Basin of Attraction (BoA) boundary.
    """

    def __init__(self, q1=None, q2=None,
                 t=1, r=3, p=0, s=5, epsilon=0.01, d=0.9,
                 func1=None, func2=None,
                 alpha=0.1, bound=1.0,
                 method="quiver", cmap=plt.cm.viridis,
                 show_trajectory=True, show_basin=True):

        self.q1 = q1
        self.q2 = q2
        self.t, self.r, self.p, self.s = t, r, p, s
        self.epsilon, self.d = epsilon, d
        self.alpha, self.bound = alpha, bound
        self.show_trajectory, self.show_basin = show_trajectory, show_basin
        self.method, self.cmap = method, cmap
        self.eps = 1e-8

        self.func1 = func1  # x-component Q-function
        self.func2 = func2  # y-component Q-function

        if self.show_trajectory:
            if self.q1 is None or self.q2 is None:
                raise ValueError("q1 and q2 must be provided if show_trajectory is True.")
            self.prepare_states()
        else:
            self.xC = self.yC = self.uC = self.vC = None
            self.xD = self.yD = self.uD = self.vD = None

        self.compute_deltas()
        self.compute_boa_curve()  # compute smooth BoA

    def prepare_states(self):
        """Computes StateC and StateD trajectories for both agents."""
        self.q1['StateC'] = self.q1['Q3'] - self.q1['Q1']
        self.q1['StateD'] = self.q1['Q4'] - self.q1['Q2']
        self.q2['StateC'] = self.q2['Q3'] - self.q2['Q1']
        self.q2['StateD'] = self.q2['Q4'] - self.q2['Q2']

        self.xC, self.yC = self.q1['StateC'].values, self.q2['StateC'].values
        self.uC, self.vC = self.xC[1:] - self.xC[:-1], self.yC[1:] - self.yC[:-1]

        self.xD, self.yD = self.q1['StateD'].values, self.q2['StateD'].values
        self.uD, self.vD = self.xD[1:] - self.xD[:-1], self.yD[1:] - self.yD[:-1]

    def compute_deltas(self):
        """Compute ΔQ for vector field."""
        Q1 = self.func1(self.t, self.r, self.p, self.s, self.epsilon, self.d)
        Q2 = self.func2(self.t, self.r, self.p, self.s, self.epsilon, self.d)

        self.dx = Q1[0][0] - Q1[0][1]
        self.dy = Q2[0][0] - Q2[0][1]

        self.Q1, self.Q2 = Q1, Q2  # store for vector fields

    def compute_boa_curve(self, num_points=100):
        """Compute smooth Basin of Attraction curve."""
        inv_scale_dx = 1 / (abs((1 - self.d) * self.dx) + self.eps)
        inv_scale_dy = 1 / (abs((1 - self.d) * self.dy) + self.eps)
        scale = min(inv_scale_dx, inv_scale_dy)

        t = np.linspace(0, 1, num_points)
        self.boa_x = -t * scale * self.dx
        self.boa_y = -t * scale * self.dy

    def field(self, X, Y, idx):
        """Vector field using row idx (0 or 1)."""
        U = self.alpha * (self.Q1[idx][0] - self.Q1[idx][1] - X)
        V = self.alpha * (self.Q2[idx][0] - self.Q2[idx][1] - Y)
        return U, V

    def add_vector_field(self, ax, field_func, x_range, y_range, density=15):
        x_vals = np.linspace(x_range[0], x_range[1], density)
        y_vals = np.linspace(y_range[0], y_range[1], density)
        X, Y = np.meshgrid(x_vals, y_vals)
        U, V = field_func(X, Y)
        ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=3)

    def plot(self):
        """Generates the vector field plot with trajectories and smooth BoA curve."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

        for i, ax in enumerate(axes):
            # vector fields from row i (0 and 1)
            self.add_vector_field(ax, lambda X, Y, idx=i: self.field(X, Y, idx),
                                  (-self.bound, self.bound), (-self.bound, self.bound))

            if self.show_trajectory:
                if i == 0:
                    ax.quiver(self.xC[:-1], self.yC[:-1], self.uC, self.vC,
                              angles='xy', scale_units='xy', scale=1,
                              color='blue', alpha=0.6, label='Trajectory (StateC)')
                else:
                    ax.quiver(self.xD[:-1], self.yD[:-1], self.uD, self.vD,
                              angles='xy', scale_units='xy', scale=1,
                              color='orange', alpha=0.6, label='Trajectory (StateD)')

            if self.show_basin:
                ax.plot(self.boa_x, self.boa_y,
                        color='red', linewidth=2, label='BoA Boundary')

            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlim(-self.bound, self.bound)
            ax.set_ylim(-self.bound, self.bound)
            ax.set_aspect('equal')
            ax.set_xlabel("Agent 1: ΔQ")
            ax.set_ylabel("Agent 2: ΔQ")
            ax.legend()
            ax.set_title(f"Vector Field {i+1}")

        plt.tight_layout()
        plt.show()


