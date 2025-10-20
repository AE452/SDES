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

# Q value difference is: (defect action - cooperate action)




def QvaluesADvsAD(t, r, p, s, epsilon, d):
    denom = 2 * (d - 1)  # equivalent to 2*(-1 + d)
    
    first_term = -((2*p - epsilon*p + epsilon*t) / denom) #defect action
    second_term = -( (2*d*p - d*epsilon*p + epsilon*r - d*epsilon*r 
                      + 2*s - 2*d*s - epsilon*s + d*epsilon*s + d*epsilon*t) / denom ) #cooperate action
    
    return np.array([[first_term, second_term], #StateC (CC)
                     [first_term, second_term]]) #StateD (CD, DC, DD)


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

#TODO: BC (Both Cooperated) and A1D (At least 1 defected)
#TODO: reduce lenght of arrows when we approach the equilibrium point

class VectorFieldAutomatic:
    """
    Vector field plotter with automatic quadrant selection based on strategy pairs.
    """

    def __init__(self, q1=None, q2=None,
                 t=1, r=0.8, p=0.2, s=0,
                 epsilon=0.1, d=0.7,
                 func1=None, func2=None,
                 strategy1=None, strategy2=None,  # Add strategy parameters
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
        
        self.strategy1 = strategy1  # Player 1 strategy
        self.strategy2 = strategy2  # Player 2 strategy (opponent)
        
        self.func1 = func1
        self.func2 = func2

        self.bound = 1 / (1 - self.delta)

        if self.show_trajectory:
            if self.q1 is None or self.q2 is None:
                raise ValueError("q1 and q2 must be provided if show_trajectory is True.")
            self.prepare_states()

    
    def get_quadrant_bounds_target(self, strategy1, strategy2, state_idx):
        """
        Determine the quadrant for plotting based on the strategies of agent1 (x-axis)
        and agent2 (y-axis).

        Strategy mapping encodes two signs:
            AD   → (+, +)   # C positive, D positive
            GT   → (-, +)   # C negative, D positive
            AGT  → (+, -)   # C positive, D negative
            AC   → (-, -)   # C negative, D negative

        Interpretation:
        - The first component corresponds to State C.
        - The second component corresponds to State D.

        Rule for quadrant selection:
        - For State C (state_idx = 0):
            * Take agent1's first sign (C) → x-axis
            * Take agent2's first sign (C) → y-axis
            → quadrant = (x_sign, y_sign)

        - For State D (state_idx = 1):
            * Take agent1's second sign (D) → x-axis
            * Take agent2's second sign (D) → y-axis
            → quadrant = (x_sign, y_sign)

        Example:
            agent1 = AD (+,+), agent2 = GT (-,+)
            - State C (idx=0): quadrant = (+, -) → lower-right
            - State D (idx=1): quadrant = (+, +) → upper-right
        """
        bound = self.bound
        quadrants = {
            ('pos', 'pos'): (0, bound, 0, bound),      # upper-right
            ('neg', 'pos'): (-bound, 0, 0, bound),     # upper-left
            ('pos', 'neg'): (0, bound, -bound, 0),     # lower-right
            ('neg', 'neg'): (-bound, 0, -bound, 0)     # lower-left
        }

        tol = 1e-10

        # Provo calcolo numerico se le funzioni sono definite
        if self.func1 is not None and self.func2 is not None:
            try:
                Qx = self.func1(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[state_idx]
                Qy = self.func2(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[state_idx]

                dx = Qx[0] - Qx[1]
                dy = Qy[0] - Qy[1]

                x_sign = 'pos' if dx > tol else ('neg' if dx < -tol else 'pos')
                y_sign = 'pos' if dy > tol else ('neg' if dy < -tol else 'pos')

                return quadrants[(x_sign, y_sign)]
            except Exception as e:
                print(f"Warning computing equilibrium numerically: {e}. Using fallback mapping.")

        # Fallback mapping strategie → (signC, signD)
        strategy_to_signs = {
            "AD": ('pos', 'pos'),
            "GT": ('neg', 'pos'),
            "AGT": ('pos', 'neg'),
            "AC": ('neg', 'neg')
        }

        s1 = strategy_to_signs.get(strategy1, ('pos', 'pos'))
        s2 = strategy_to_signs.get(strategy2, ('pos', 'pos'))

        # Entrambi prendono il segno relativo allo stesso state_idx
        sx = s1[state_idx]  # agent1 → x
        sy = s2[state_idx]  # agent2 → y

        return quadrants[(sx, sy)]

    
    def get_quadrant_bounds_case(self, strategy1, strategy2, state_idx):
        """
        select the quadrant
        """
        bound = self.bound
        quadrants = {
            ('pos', 'pos'): (0, bound, 0, bound),      # upper-right
            ('neg', 'pos'): (-bound, 0, 0, bound),     # upper-left
            ('pos', 'neg'): (0, bound, -bound, 0),     # lower-right
            ('neg', 'neg'): (-bound, 0, -bound, 0)     # lower-left
        }

        

        # Fallback mapping strategie → (signC, signD)
        strategy_to_signs = {
            "AD": ('pos', 'pos'),
            "GT": ('neg', 'pos'),
            "AGT": ('pos', 'neg'),
            "AC": ('neg', 'neg')
        }

        s1 = strategy_to_signs.get(strategy1)
        s2 = strategy_to_signs.get(strategy2)

        
        sx = s1[state_idx]  # agent1 → x
        sy = s2[state_idx]  # agent2 → y

        return quadrants[(sx, sy)]


    def prepare_states(self):
        """Calcola StateC e StateD per le traiettorie."""
        self.q1['StateBC'] = self.q1['Q3'] - self.q1['Q1']
        self.q1['StateA1D'] = self.q1['Q4'] - self.q1['Q2']
        self.q2['StateBC'] = self.q2['Q3'] - self.q2['Q1']
        self.q2['StateA1D'] = self.q2['Q4'] - self.q2['Q2']

        self.xC, self.yC = self.q1['StateBC'].values, self.q2['StateBC'].values
        self.uC, self.vC = self.xC[1:] - self.xC[:-1], self.yC[1:] - self.yC[:-1]

        self.xD, self.yD = self.q1['StateA1D'].values, self.q2['StateA1D'].values
        self.uD, self.vD = self.xD[1:] - self.xD[:-1], self.yD[1:] - self.yD[:-1]

    def compute_boa(self, idx):
        """Calculation of the basin of attraction boundary."""
        if self.func1 is None or self.func2 is None:
            return None, None

        Q1 = self.func1(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[idx]
        Q2 = self.func2(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[idx]

        dx = Q1[0] - Q1[1]
        dy = Q2[0] - Q2[1]

        scale = min(1 / ((1 - self.delta) * abs(dx) + 1e-8),
                    1 / ((1 - self.delta) * abs(dy) + 1e-8))
        boa_x = np.array([0, -scale * dx])
        boa_y = np.array([0, -scale * dy])
        return boa_x, boa_y

    def add_vector_field_old(self, ax, xmin, xmax, ymin, ymax, func1, func2, idx):
        """Add vector field to the specified region."""
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
        
    def add_vector_field(self, ax, xmin, xmax, ymin, ymax, func1, func2, idx):
        X_vals = np.linspace(xmin, xmax, self.density)
        Y_vals = np.linspace(ymin, ymax, self.density)
        X, Y = np.meshgrid(X_vals, Y_vals)
        
        # Compute ΔQ
        Q1 = func1(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[idx]
        Q2 = func2(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[idx]
        dQx = Q1[0] - Q1[1]
        dQy = Q2[0] - Q2[1]

        U = self.alpha * (dQx - X)
        V = self.alpha * (dQy - Y)

        # Compute magnitude
        M = np.sqrt(U**2 + V**2)

        # --- Scale arrows to shrink near equilibrium ---
        # Instead of normalizing, compress magnitudes smoothly
        scale_factor = np.sqrt(M) / (1 + np.sqrt(M))  
        U = U / (M + 1e-8) * scale_factor/3
        V = V / (M + 1e-8) * scale_factor/3

        # Plot vector field
        ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1,
                color='black', alpha=0.7, width=0.003)


    def plot(self, show_all_quadrants=False):
        """Plot two subplots with either quadrant-specific or full-plane vector fields."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        subplots = [
            ('StateBC', 0),   # state index 0
            ('StateA1D', 1)   # state index 1
        ]

        for i, (state_name, idx) in enumerate(subplots):
            ax = axes[i]

            if show_all_quadrants:
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
            else:
                # Restrict to the quadrant selected by strategies
                xmin, xmax, ymin, ymax = self.get_quadrant_bounds_case(
                    self.strategy1, self.strategy2, idx
                )

                # Add vector field
                self.add_vector_field(ax, xmin, xmax, ymin, ymax, self.func1, self.func2, idx)

            # Trajectory
            if self.show_trajectory:
                x = getattr(self, f'x{state_name[-1]}')
                y = getattr(self, f'y{state_name[-1]}')
                u = getattr(self, f'u{state_name[-1]}')
                v = getattr(self, f'v{state_name[-1]}')
                color = 'blue' if state_name == 'StateBC' else 'orange'
                ax.quiver(
                    x[:-1], y[:-1], u, v,
                    angles='xy', scale_units='xy', scale=1,
                    color=color, alpha=0.8, width=0.004,
                    label=f'Trajectory {state_name}'
                )
                #this is a red cross (x) plot at the end of the trajectory, we decided to eliminate 
                #ax.scatter(x[-1], y[-1], color='red', marker='X', s=50)

            # Basin of attraction
            if self.show_basin:
                boa_x, boa_y = self.compute_boa(idx)
                if boa_x is not None and boa_y is not None:
                    ax.plot(boa_x, boa_y, color='red', linewidth=2, label='BoA Boundary')

            if show_all_quadrants:
                bound = self.bound
                ax.set_xlim(-bound, bound)
                ax.set_ylim(-bound, bound)
            else:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)


            # Styling
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
            ax.set_aspect('equal')
            ax.set_xlabel(f'Agent 1 ΔQ ({state_name})')
            ax.set_ylabel(f'Agent 2 ΔQ ({state_name})')
            ax.set_title(f'{state_name}: {self.strategy1} vs {self.strategy2}')
            ax.grid(True, alpha=0.2)
            if self.show_trajectory or self.show_basin:
                ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()
