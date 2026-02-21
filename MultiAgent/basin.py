import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
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
    """Q-values for AllD vs AllD"""
    denom = 4 * (-1 + d)
    
    q11 = (-((-2 + epsilon) * (-2 + d * epsilon) * p) + 
           epsilon * (-2 * d * s - 2 * t + d * epsilon * (-r + s + t))) / denom
    
    q12 = -(2 * epsilon * (r - s) + 4 * s + 
            d * (-2 + epsilon) * ((-2 + epsilon) * p + 2 * s + epsilon * (r - s - t))) / denom
    
    return np.array([[q11, q12],
                     [q11, q12]])


def QvaluesADvsAGT(t, r, p, s, epsilon, d):
    """Q-values for AllD vs GTFT"""
    denom = 2 * (-1 + d) * (-2 + d * (-1 + epsilon) * epsilon)
    
    q11 = ((-2 + epsilon) * (-2 + d * (2 + (-2 + epsilon) * epsilon)) * p + 
           d * epsilon * (-((-2 + epsilon) * r) + epsilon * (1 + d - d * epsilon) * s) + 
           (2 * epsilon + d * (4 + epsilon * (-6 + (2 + d * (-1 + epsilon) - epsilon) * epsilon))) * t) / denom
    
    q12 = (2 * epsilon * (r - s) + 4 * s - 
           d**2 * (-2 + epsilon) * (-1 + epsilon) * epsilon * (s - t) + 
           d * (-4 * s + 4 * t + epsilon * ((-2 + epsilon)**2 * p + 4 * s - 6 * t - 
                                             epsilon * (r + s + (-4 + epsilon) * t)))) / denom
    
    q21 = (epsilon * ((2 + 2 * d**2 * (-1 + epsilon)**2 - d * epsilon**2) * p - 
                      d * (-2 + epsilon) * r + d * epsilon * (1 + d - d * epsilon) * s) - 
           (-2 + epsilon) * (2 + d * (d * (-1 + epsilon) - epsilon) * epsilon) * t) / denom
    
    q22 = (-2 * (-2 + epsilon) * r + 2 * epsilon * s + 
           d * (-((-2 + epsilon) * ((2 + (-2 + epsilon) * epsilon) * p + (-2 + epsilon) * r)) - 
                epsilon**2 * s + (-2 + epsilon)**2 * epsilon * t) + 
           d**2 * (-2 + epsilon) * (-1 + epsilon) * (2 * (-1 + epsilon) * p + 2 * t - 
                                                      epsilon * (s + t))) / denom
    
    return np.array([[q11, q12],
                     [q21, q22]])


def QvaluesADvsAC(t, r, p, s, epsilon, d):
    """Q-values for AllD vs AllC"""
    denom = 4 * (-1 + d)
    
    q11 = (d * epsilon**2 * (p + r - s - t) - 4 * t + 
           2 * epsilon * (-p - d * r + t + d * t)) / denom
    
    q12 = (2 * (-2 + epsilon) * r - 2 * epsilon * s + 
           d * (-2 + epsilon) * (-2 * r + epsilon * (p + r - s - t) + 2 * t)) / denom
    
    return np.array([[q11, q12],
                     [q11, q12]])


def QvaluesGTvsGT(t, r, p, s, epsilon, d):
    """Q-values for GTFT vs GTFT"""
    denom = 4 * (-1 + d) * (1 + d * (-1 + epsilon))
    
    q11 = (-2 * epsilon * p + 2 * (-2 + epsilon) * t - 
           d**2 * (-2 + epsilon) * (-1 + epsilon) * (2 * (-1 + epsilon) * p + 2 * t - 
                                                      epsilon * (s + t)) + 
           d * ((-4 + (8 - 3 * epsilon) * epsilon) * p + 8 * t + 
                epsilon * (epsilon * (-r + s + 3 * t) - 2 * (s + 5 * t)))) / denom
    
    q12 = (2 * (-2 + epsilon) * r - 2 * epsilon * s + 
           d * (4 * r + epsilon**2 * (p - r - s + t) - 2 * epsilon * (p + r - s + t)) - 
           d**2 * (-1 + epsilon) * epsilon * (2 * (-1 + epsilon) * p + 2 * t - 
                                               epsilon * (s + t))) / denom
    
    q21 = ((2 * (-2 + epsilon) + d * (4 + epsilon * (-4 - 2 * d * (-1 + epsilon)**2 + epsilon))) * p + 
           d * epsilon * (-epsilon * r + (1 + d * (-1 + epsilon)) * (-2 + epsilon) * s) + 
           (1 + d * (-1 + epsilon)) * epsilon * (-2 + d * epsilon) * t) / denom
    
    q22 = (-2 * epsilon * r + 2 * (-2 + epsilon) * s - 
           d**2 * (-2 + epsilon) * (-1 + epsilon) * (2 * (-1 + epsilon) * p + 2 * s - 
                                                      epsilon * (s + t)) + 
           d * ((-4 - 3 * (-2 + epsilon) * epsilon) * p + 
                (-2 + epsilon) * (-4 * s + epsilon * (-r + 3 * s + t)))) / denom
    
    return np.array([[q11, q12],
                     [q21, q22]])

def calculate_boa_boundary_ADGT(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6, state_idx=0):
    """
    Calculate basin-of-attraction boundary vectors for the (AD, GT) configuration.
    
    This matches the compute_boa method in VectorFieldAutomatic class.
    The boundary is determined by the policy gradient differences scaled by (1-δ).
    
    Formula: 
        scale = min(1/((1-δ)|ΔQx|), 1/((1-δ)|ΔQy|))
        boundary = [(-scale·ΔQx, -scale·ΔQy)]
    
    where ΔQx = Q^AD(0) − Q^AD(1) and ΔQy = Q^GT(0) − Q^GT(1)

    Parameters
    ----------
    t, r, p, s : float
        Prisoner's dilemma payoffs (temptation, reward, punishment, sucker).
    epsilon : float
        Exploration parameter.
    d : float
        Discount factor (delta).
    state_idx : int
        State index: 0 = StateBC (CC), 1 = StateA1D.

    Returns
    -------
    tuple of np.ndarray
        (boa_x, boa_y): Basin boundary vectors, each array([0, boundary_value])
    """
    Q1 = QvaluesADvsAD(t, r, p, s, epsilon, d)[state_idx]
    Q2 = QvaluesGTvsGT(t, r, p, s, epsilon, d)[state_idx]
    
    dx = Q1[0] - Q1[1]
    dy = Q2[0] - Q2[1]
    
    # Scale factor matching VectorFieldAutomatic.compute_boa
    scale = min(1 / ((1 - d) * abs(dx) + 1e-8),
                1 / ((1 - d) * abs(dy) + 1e-8))
    
    boa_x = np.array([0, -scale * dx])
    boa_y = np.array([0, -scale * dy])
    
    return boa_x, boa_y

def calculate_pi_ADGT(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6, state_idx=0):
    """
    Calculate basin probability from paper formula.
    
    π = (1/2) × min(|ΔQx|, |ΔQy|) / max(|ΔQx|, |ΔQy|)
    
    This π represents the basin of whichever strategy has the SMALLER gradient.
    """
    Q1 = QvaluesADvsAD(t, r, p, s, epsilon, d)[state_idx]
    Q2 = QvaluesGTvsGT(t, r, p, s, epsilon, d)[state_idx]
    
    dx = Q1[0] - Q1[1]
    dy = Q2[0] - Q2[1]
    
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    
    if np.isclose(abs_dx, 0.0) and np.isclose(abs_dy, 0.0):
        return 0.5  # Equal basins
    if np.isclose(abs_dx, 0.0):
        return 0.0  # Only GT has gradient → P(GT)=1, P(AD)=0
    if np.isclose(abs_dy, 0.0):
        return 1.0  # Only AD has gradient → P(AD)=1, P(GT)=0
    
    # π = basin of smaller gradient
    pi_small = 0.5 * min(abs_dx, abs_dy) / max(abs_dx, abs_dy)
    
    # Determine which strategy has smaller gradient
    if abs_dx < abs_dy:
        # AD has smaller gradient → π represents P(AD)
        P_AD = pi_small
    else:
        # GT has smaller gradient → π represents P(GT), so P(AD) = 1 - π
        P_AD = 1.0 - pi_small
    
    return float(np.clip(P_AD, 0.0, 1.0))


def theoretical_P_AD_from_ADGT(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6, 
                                state_idx=0, delta=None, e=None):
    """
    Theoretical P(AD) from the (AD, GT) configuration.
    Uses the paper formula with correct interpretation.
    """
    if delta is not None:
        d = delta
    if e is not None:
        epsilon = e
    
    P_AD = calculate_pi_ADGT(t, r, p, s, epsilon=epsilon, d=d, state_idx=state_idx)
    return float(P_AD)

"""
Table printing functions for basin of attraction analysis.
Add these to your existing notebook that already has the Q-value functions.
"""
def get_aggregate_bounds(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6, state_idx=0, P_AD = True):
    Q_AD = QvaluesADvsAD(t, r, p, s, epsilon, d)[state_idx]
    Q_GT = QvaluesGTvsGT(t, r, p, s, epsilon, d)[state_idx]
    
    abs_dx = abs(Q_AD[0] - Q_AD[1])
    abs_dy = abs(Q_GT[0] - Q_GT[1])
    
    pi_small = 0.5 * min(abs_dx, abs_dy) / max(abs_dx, abs_dy) if max(abs_dx, abs_dy) > 0 else 0.5
    pi_GTGT = (1.0 - pi_small) if abs_dx < abs_dy else (pi_small if abs_dx > abs_dy else 0.5)
    if P_AD:
        return  1 - (1/16) * (7 + 3 * pi_GTGT), 1 - (1/16) * (2 + 3 * pi_GTGT)
    return (1/16) * (2 + 3 * pi_GTGT), (1/16) * (7 + 3 * pi_GTGT)

def print_paper_table(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6, state_idx=0):
    """
    Print one configuration in paper table style.
    Uses your existing Q-value and basin calculation functions.
    """
    # Calculate gradients using your existing functions
    Q_AD = QvaluesADvsAD(t, r, p, s, epsilon, d)[state_idx]
    Q_GT = QvaluesGTvsGT(t, r, p, s, epsilon, d)[state_idx]
    
    dx = Q_AD[0] - Q_AD[1]
    dy = Q_GT[0] - Q_GT[1]
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    
    # Calculate π values
    pi_small = 0.5 * min(abs_dx, abs_dy) / max(abs_dx, abs_dy) if max(abs_dx, abs_dy) > 0 else 0.5
    
    if abs_dx < abs_dy:
        pi_ADAD = pi_small
        pi_GTGT = 1.0 - pi_small
        basin_analysis = "|ΔQx| < |ΔQy|"
        adad_label = "(weaker)"
        gtgt_label = "(stronger)"
    elif abs_dx > abs_dy:
        pi_GTGT = pi_small
        pi_ADAD = 1.0 - pi_small
        basin_analysis = "|ΔQy| < |ΔQx|"
        adad_label = "(stronger)"
        gtgt_label = "(weaker)"
    else:
        pi_GTGT = pi_ADAD = 0.5
        basin_analysis = "|ΔQx| = |ΔQy|"
        adad_label = gtgt_label = ""
    
    # Calculate conditional probabilities
    P_GTGT_given_GTGT = 1.0
    P_GTGT_given_ADGT = pi_GTGT
    P_GTGT_given_GTAC = 0.5 * (1 + pi_GTGT)
    
    # Calculate bounds
    lower_bound = (1/16) * (2 + 3 * pi_GTGT)
    upper_bound = (1/16) * (7 + 3 * pi_GTGT)
    
    # Print
    print(f"Payoffs: R = {r}, P = {p}")
    print(f"{'-'*70}")
    print()
    print("Q-value gradients:")
    print(f"  |ΔQx| (AD gradient)       {abs_dx:.3f}")
    print(f"  |ΔQy| (GT gradient)       {abs_dy:.3f}")
    print()
    print(f"Basin analysis:             {basin_analysis}")
    print(f"  π_small                   {pi_small:.3f}")
    print(f"  π_ADAD {adad_label:<17} {pi_ADAD:.3f}")
    
    gtgt_str = f"  π_GTGT {gtgt_label:<17} {pi_GTGT:.3f}"
    if "(stronger)" in gtgt_label:
        gtgt_str += "  ★"
    print(gtgt_str)
    print()
    
    print("Conditional probabilities:")
    print(f"  P(GT,GT | GT,GT)          {P_GTGT_given_GTGT:.4f}")
    print(f"  P(GT,GT | AD,GT)          {P_GTGT_given_ADGT:.4f}")
    print(f"  P(GT,GT | GT,AC)          {P_GTGT_given_GTAC:.4f}")
    print()
    
    print("Aggregate bounds:")
    print(f"  Lower bound               {lower_bound:.3f}")
    print(f"  Upper bound               {upper_bound:.3f}")
    print()
    return lower_bound, upper_bound

def print_side_by_side_tables(configs: list):
    """
    Print two configurations side by side.
    
    Args:
        configs: List of tuples (t, r, p, s, epsilon, d, state_idx)
                 Example: [(1, 0.7, 0.2, 0, 0.1, 0.6, 0), (1, 0.8, 0.5, 0, 0.1, 0.6, 0)]
    """
    if len(configs) != 2:
        raise ValueError("Need exactly 2 configurations")
    
    results = []
    for t, r, p, s, epsilon, d, state_idx in configs:
        # Calculate gradients
        Q_AD = QvaluesADvsAD(t, r, p, s, epsilon, d)[state_idx]
        Q_GT = QvaluesGTvsGT(t, r, p, s, epsilon, d)[state_idx]
        
        dx = Q_AD[0] - Q_AD[1]
        dy = Q_GT[0] - Q_GT[1]
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        # Calculate π values
        pi_small = 0.5 * min(abs_dx, abs_dy) / max(abs_dx, abs_dy) if max(abs_dx, abs_dy) > 0 else 0.5
        
        if abs_dx < abs_dy:
            pi_ADAD = pi_small
            pi_GTGT = 1.0 - pi_small
            basin_analysis = "|ΔQx| < |ΔQy|"
            adad_label = "(weaker)"
            gtgt_label = "(stronger)"
        elif abs_dx > abs_dy:
            pi_GTGT = pi_small
            pi_ADAD = 1.0 - pi_small
            basin_analysis = "|ΔQy| < |ΔQx|"
            adad_label = "(stronger)"
            gtgt_label = "(weaker)"
        else:
            pi_GTGT = pi_ADAD = 0.5
            basin_analysis = "|ΔQx| = |ΔQy|"
            adad_label = gtgt_label = ""
        
        results.append({
            'r': r, 'p': p,
            'abs_dx': abs_dx, 'abs_dy': abs_dy,
            'basin_analysis': basin_analysis,
            'pi_small': pi_small, 'pi_ADAD': pi_ADAD, 'pi_GTGT': pi_GTGT,
            'adad_label': adad_label, 'gtgt_label': gtgt_label,
            'P_GTGT_given_GTGT': 1.0,
            'P_GTGT_given_ADGT': pi_GTGT,
            'P_GTGT_given_GTAC': 0.5 * (1 + pi_GTGT),
            'lower_bound': (1/16) * (2 + 3 * pi_GTGT),
            'upper_bound': (1/16) * (7 + 3 * pi_GTGT)
        })
    
    # Print headers
    w = 68
    print(f"{'='*140}")
    print(f"{'Basin of Attraction Analysis':^140}")
    print(f"{'='*140}\n")
    print(f"{'Payoffs: R = ' + str(results[0]['r']) + ', P = ' + str(results[0]['p']):<{w}} | Payoffs: R = {results[1]['r']}, P = {results[1]['p']}")
    print(f"{'-'*w} | {'-'*w}\n")
    
    # Q-value gradients
    print(f"{'Q-value gradients:':<{w}} | Q-value gradients:")
    print(f"{'  |ΔQx| (AD gradient)       ' + f'{results[0]['abs_dx']:.3f}':<{w}} |   |ΔQx| (AD gradient)       {results[1]['abs_dx']:.3f}")
    print(f"{'  |ΔQy| (GT gradient)       ' + f'{results[0]['abs_dy']:.3f}':<{w}} |   |ΔQy| (GT gradient)       {results[1]['abs_dy']:.3f}\n")
    
    # Basin analysis
    print(f"{'Basin analysis:             ' + results[0]['basin_analysis']:<{w}} | Basin analysis:             {results[1]['basin_analysis']}")
    print(f"{'  π_small                   ' + f'{results[0]['pi_small']:.3f}':<{w}} |   π_small                   {results[1]['pi_small']:.3f}")
    print(f"{'  π_ADAD ' + results[0]['adad_label'] + ' ' * (11 - len(results[0]['adad_label'])) + f'{results[0]['pi_ADAD']:.3f}':<{w}} |   π_ADAD {results[1]['adad_label']}{' ' * (11 - len(results[1]['adad_label']))}{results[1]['pi_ADAD']:.3f}")
    
    left_gt = f"  π_GTGT {results[0]['gtgt_label']}{' ' * (11 - len(results[0]['gtgt_label']))}{results[0]['pi_GTGT']:.3f}"
    if "(stronger)" in results[0]['gtgt_label']:
        left_gt += "  ★"
    right_gt = f"  π_GTGT {results[1]['gtgt_label']}{' ' * (11 - len(results[1]['gtgt_label']))}{results[1]['pi_GTGT']:.3f}"
    if "(stronger)" in results[1]['gtgt_label']:
        right_gt += "  ★"
    print(f"{left_gt:<{w}} | {right_gt}\n")
    
    # Conditional probabilities
    print(f"{'Conditional probabilities:':<{w}} | Conditional probabilities:")
    print(f"{'  P(GT,GT | GT,GT)          ' + f'{results[0]['P_GTGT_given_GTGT']:.4f}':<{w}} |   P(GT,GT | GT,GT)          {results[1]['P_GTGT_given_GTGT']:.4f}")
    print(f"{'  P(GT,GT | AD,GT)          ' + f'{results[0]['P_GTGT_given_ADGT']:.4f}':<{w}} |   P(GT,GT | AD,GT)          {results[1]['P_GTGT_given_ADGT']:.4f}")
    print(f"{'  P(GT,GT | GT,AC)          ' + f'{results[0]['P_GTGT_given_GTAC']:.4f}':<{w}} |   P(GT,GT | GT,AC)          {results[1]['P_GTGT_given_GTAC']:.4f}\n")
    
    # Aggregate bounds
    print(f"{'Aggregate bounds:':<{w}} | Aggregate bounds:")
    print(f"{'  Lower bound               ' + f'{results[0]['lower_bound']:.3f}':<{w}} |   Lower bound               {results[1]['lower_bound']:.3f}")
    print(f"{'  Upper bound               ' + f'{results[0]['upper_bound']:.3f}':<{w}} |   Upper bound               {results[1]['upper_bound']:.3f}\n")
    print(f"{'='*140}")

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
            # Use LaTeX for delta symbol (\Delta)
            ax.set_xlabel(f'Agent 1 $\\Delta$Q ({state_name})')
            ax.set_ylabel(f'Agent 2 $\\Delta$Q ({state_name})')
            ax.set_title(f'{state_name}: {self.strategy1} vs {self.strategy2}')
            ax.grid(True, alpha=0.2)
            if self.show_trajectory or self.show_basin:
                try:
                    ax.legend(fontsize=8)
                except:
                    pass  # Skip legend if there's an issue

        plt.tight_layout()
        # In Jupyter notebooks, figures are displayed automatically at the end of cell execution
        # Don't call plt.show() as it can cause display issues
        # The figure will be shown automatically by Jupyter
