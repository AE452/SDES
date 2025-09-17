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

# --------------------------------------------------------
# How the automatic quadrant determination based on ΔQ values works:
#
# The strategy-to-quadrant mapping (AD → (+,+), GT → (-,+), 
# AGT → (+,-), AC → (-,-)) gives a geometric intuition of 
# where each pure strategy lies in the ΔQ-plane.
#
# To automatically select the quadrant for plotting a given 
# strategy pair and state, we compute the actual ΔQ values:
#   - Player 1 (x-axis): dx = Qx[0] - Qx[1] (defect - cooperate)
#   - Player 2 (y-axis): dy = Qy[0] - Qy[1]
#
# Then assign signs:
#   - x_sign = 'pos' if dx >= 0 else 'neg'
#   - y_sign = 'pos' if dy >= 0 else 'neg'
#
# The quadrant is determined from the combination of signs:
#   ('pos','pos') → upper right
#   ('neg','pos') → upper left
#   ('pos','neg') → lower right
#   ('neg','neg') → lower left
#
# This approach ensures that the vector field and trajectory
# initialization reflect the actual Q-value dynamics for
# the selected strategy pair and state, making the plotting
# fully automatic and robust.
# --------------------------------------------------------

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

    def get_quadrant_bounds(self, strategy1, strategy2, state_idx):
        """
        Determina il quadrante valutando i segni delle differenze di Q
        per x (player1) e y (player2) nello stato `state_idx`.
        Ritorna (xmin, xmax, ymin, ymax).
        """
        bound = self.bound
        quadrants = {
            ('pos', 'pos'): (0, bound, 0, bound),      # upper-right
            ('neg', 'pos'): (-bound, 0, 0, bound),     # upper-left
            ('pos', 'neg'): (0, bound, -bound, 0),     # lower-right
            ('neg', 'neg'): (-bound, 0, -bound, 0)     # lower-left
        }

        tol = 1e-10  # tolleranza per evitare casi numerici ambigui

        # Se ho le due funzioni, uso i valori numerici (metodo robusto)
        if self.func1 is not None and self.func2 is not None:
            try:
                # Nota: func1 -> produce Q per l'asse x (difetto-coop per player1),
                #       func2 -> produce Q per l'asse y (difetto-coop per player2).
                Qx = self.func1(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[state_idx]
                Qy = self.func2(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[state_idx]

                dx = Qx[0] - Qx[1]   # definizione tua: (defect - cooperate)
                dy = Qy[0] - Qy[1]

                x_sign = 'pos' if dx > tol else ('neg' if dx < -tol else 'pos')  # tie -> pos
                y_sign = 'pos' if dy > tol else ('neg' if dy < -tol else 'pos')

                return quadrants[(x_sign, y_sign)]
            except Exception as e:
                # fallback se qualcosa fallisce nel calcolo numerico
                print(f"Warning computing equilibrium numerically: {e}. Using fallback mapping.")

        # Se non ho le funzioni (o il calcolo è fallito), uso una semplice regola basata
        # sulle etichette delle strategie (meno preferibile ma come fallback)
        strategy_to_sign = {
            "AD": ('pos', 'pos'),
            "GT": ('neg', 'pos'),
            "AGT": ('pos', 'neg'),
            "AC": ('neg', 'neg')
        }

        sx = strategy_to_sign.get(strategy1, ('pos','pos'))[0]  # prima componente per x
        sy = strategy_to_sign.get(strategy2, ('pos','pos'))[1]  # seconda componente per y

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

    def add_vector_field(self, ax, xmin, xmax, ymin, ymax, func1, func2, idx):
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

    def plot(self):
        """Plot two subplots with quadrant-specific vector fields."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        subplots = [
            ('StateC', 0),  # first array
            ('StateD', 1)   # second array
        ]

        for i, (state_name, idx) in enumerate(subplots):
            ax = axes[i]

            # Get the appropriate quadrant for this strategy pair and state
            xmin, xmax, ymin, ymax = self.get_quadrant_bounds(
                self.strategy1, self.strategy2, idx
            )
            
            # Add vector field only to the selected quadrant
            self.add_vector_field(ax, xmin, xmax, ymin, ymax, self.func1, self.func2, idx)

            # Trajectory
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
            if self.show_basin:
                boa_x, boa_y = self.compute_boa(idx)
                if boa_x is not None and boa_y is not None:
                    ax.plot(boa_x, boa_y, color='red', linewidth=2, label='BoA Boundary')

            # Set plot limits to the selected quadrant
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            
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
