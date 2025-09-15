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


class VectorFieldAutomatic:
    """
    Vector field plotter automatico come Mathematica:
    - ΔQ calcolato per ogni punto della meshgrid
    - Gestisce automaticamente uno o più equilibri
    - Due subplots: StateC-StateC e StateD-StateD
    - Traiettorie e BoA incluse
    """

    def __init__(self, q1=None, q2=None,
                 t=1, r=0.8, p=0.2, s=0,
                 epsilon=0.1, d=0.7,
                 func1=None, func2=None,
                 alpha=0.3, density=15,
                 show_trajectory=True, show_basin=True):
        
        self.q1 = q1
        self.q2 = q2
        self.t, self.r, self.p, self.s = t, r, p, s
        self.epsilon = epsilon
        self.delta = d
        self.alpha = alpha
        self.density = density
        self.show_trajectory = show_trajectory
        self.show_basin = show_basin

        self.func1 = func1
        self.func2 = func2

        self.bound = 1 / (1 - self.delta)

        if self.show_trajectory:
            if self.q1 is None or self.q2 is None:
                raise ValueError("q1 and q2 must be provided if show_trajectory is True.")
            self.prepare_states()

        self.compute_boa()

    def prepare_states(self):
        """Calcola StateC e StateD per le traiettorie."""
        self.q1['StateC'] = self.q1['Q3'] - self.q1['Q1']
        self.q1['StateD'] = self.q1['Q4'] - self.q1['Q2']
        self.q2['StateC'] = self.q2['Q3'] - self.q2['Q1']
        self.q2['StateD'] = self.q2['Q4'] - self.q2['Q2']

        self.xC, self.yC = self.q1['StateC'].values, self.q2['StateC'].values
        self.uC, self.vC = self.xC[1:] - self.xC[:-1], self.yC[1:] - self.yC[:-1]

        self.xD, self.yD = self.q1['StateD'].values, self.q2['StateD'].values
        self.uD, self.vD = self.xD[1:] - self.xD[:-1], self.yD[1:] - self.yD[:-1]

    def compute_boa(self):
        """Calcola BoA (basin of attraction) dalle funzioni Q in (0,0)."""
        if self.func1 is None or self.func2 is None:
            self.boa_x = self.boa_y = None
            return

        Q1 = self.func1(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[0]
        Q2 = self.func2(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[0]

        dx = Q1[0] - Q1[1]
        dy = Q2[0] - Q2[1]

        scale = min(1 / ((1 - self.delta) * abs(dx) + 1e-8),
                    1 / ((1 - self.delta) * abs(dy) + 1e-8))
        self.boa_x = np.array([0, -scale * dx])
        self.boa_y = np.array([0, -scale * dy])

    def vector_field(self, X, Y, func, idx):
        """
        Calculate ΔQ for each point (X,Y) using the function Q.
        """
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        if func is None:
            return U, V

        Q = func(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[idx]

        dQx = Q[0] - Q[1]
        dQy = Q[0] - Q[1]

        U = self.alpha * (dQx - X)
        V = self.alpha * (dQy - Y)

        mag = np.sqrt(U**2 + V**2) + 1e-8
        return U / mag, V / mag

    def add_vector_field(self, ax, xmin, xmax, ymin, ymax, func1, func2, idx):
        X_vals = np.linspace(xmin, xmax, self.density)
        Y_vals = np.linspace(ymin, ymax, self.density)
        X, Y = np.meshgrid(X_vals, Y_vals)
        
        # Compute vector field: ΔQx from func1, ΔQy from func2
        Q1 = func1(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[idx]
        Q2 = func2(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[idx]
        
        dQx = Q1[0] - Q1[1]
        dQy = Q2[0] - Q2[1]

        U = self.alpha * (dQx - X)
        V = self.alpha * (dQy - Y)

        mag = np.sqrt(U**2 + V**2) + 1e-8
        ax.quiver(X, Y, U/mag, V/mag, angles='xy', scale_units='xy', scale=12,
                color='black', alpha=0.7, width=0.003)


    def plot(self):
        """Design two subplots: StateC-StateC e StateD-StateD."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

        subplots = [
            ('StateC', 0),  # first array
            ('StateD', 1)   # second array
        ]

        for i, (state_name, idx) in enumerate(subplots):
            ax = axes[i]

            # Quadrants
            bound = self.bound
            quadrants = [
                (0, bound, -bound, 0),
                (-bound, 0, -bound, 0),
                (0, bound, 0, bound),
                (-bound, 0, 0, bound)
            ]
            for xmin, xmax, ymin, ymax in quadrants:
                # Use both func1 and func2 for each subplot with correct idx
                self.add_vector_field(ax, xmin, xmax, ymin, ymax, self.func1, self.func2, idx)

            # Traiettoria
            if self.show_trajectory:
                x = getattr(self, f'x{state_name[-1]}')
                y = getattr(self, f'y{state_name[-1]}')
                u = getattr(self, f'u{state_name[-1]}')
                v = getattr(self, f'v{state_name[-1]}')
                color = 'blue' if state_name == 'StateC' else 'orange'
                ax.quiver(x[:-1], y[:-1], u, v, angles='xy', scale_units='xy', scale=1,
                        color=color, alpha=0.8, width=0.004, label=f'Trajectory {state_name}')
                ax.scatter(x[-1], y[-1], color='red', marker='X', s=50)

            # BoA
            if self.show_basin and self.boa_x is not None:
                ax.plot(self.boa_x, self.boa_y, color='red', linewidth=2, label='BoA Boundary')

            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-bound, bound)
            ax.set_aspect('equal')
            ax.set_xlabel(f'Agent 1 ΔQ ({state_name})')
            ax.set_ylabel(f'Agent 2 ΔQ ({state_name})')
            ax.set_title(f'{state_name} vs {state_name}')
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()
