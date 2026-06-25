import os
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


# ---------------------------------------------------------------------------
# Appendix target Δq helpers (paper app:QvalDiff) and Prop. 17 quantities
# Convention: Δq = q[D] - q[C]; state_idx 0 = BC, 1 = A1D; d ≡ γ.
# ---------------------------------------------------------------------------

def target_delta_q_AD(t, r, p, s, epsilon, d):
    """Δq^AD (same in BC and A1D)."""
    return 0.5 * (2 * p - 2 * s + epsilon * (t + s - r - p))


def target_delta_q_AC(t, r, p, s, epsilon, d):
    """Δq^AC (same in BC and A1D)."""
    return (t - r) + 0.5 * epsilon * (p + r - s - t)


def target_delta_q_GT(t, r, p, s, epsilon, d, state_idx=0):
    """Δq^GT in BC (0) or A1D (1)."""
    if state_idx == 0:
        num = (2 * t - 2 * r + epsilon * (p + r - s - t)
               - d * (1 - epsilon) * (2 * t - 2 * p - epsilon * (s + t - 2 * p)))
        den = 2 - 2 * d * (1 - epsilon)
    else:
        num = ((2 - 2 * d * (1 - epsilon) ** 2) * p - 2 * (1 - d) * s
               + epsilon * (t + s - r - p - d * (3 - epsilon) * s - d * (1 - epsilon) * t))
        den = 2 - 2 * d * (1 - epsilon)
    return num / den


def _resolve_params(d=None, delta=None, epsilon=None, e=None):
    gamma = d if d is not None else delta
    eps = epsilon if epsilon is not None else e
    if gamma is None or eps is None:
        raise ValueError("Provide discount (d or delta) and exploration (epsilon or e)")
    return gamma, eps


def delta_q_from_qvalues(opponent, state_idx, t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6, player_row=0):
    """Map opponent strategy to Qvalues* Δq for a given player row."""
    func = {"AD": QvaluesADvsAD, "GT": QvaluesGTvsGT, "AC": QvaluesADvsAC, "AGT": QvaluesADvsAGT}[opponent]
    q = func(t, r, p, s, epsilon, d)[player_row]
    return float(q[0] - q[1])


def calculate_B_hat(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6, state_idx=0,
                    delta=None, e=None):
    """
    BoA(π^GT | (AD, GT)) from eq. Bhat in paper.tex.

    Uses Δq_1^GT(BC) and Δq_2^AD(BC).
    """
    gamma, eps = _resolve_params(d=d, delta=delta, epsilon=epsilon, e=e)
    g1 = abs(target_delta_q_GT(t, r, p, s, eps, gamma, state_idx=state_idx))
    g2 = abs(target_delta_q_AD(t, r, p, s, eps, gamma))
    if np.isclose(g1, 0.0) and np.isclose(g2, 0.0):
        return 0.5
    if np.isclose(g2, 0.0):
        return 1.0
    if np.isclose(g1, 0.0):
        return 0.0
    if g2 >= g1:
        return 0.5 * g1 / g2
    return 1.0 - 0.5 * g2 / g1


def calculate_C_hat(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6,
                    delta=None, e=None):
    """
    Lower bound ĉ = P[T_A1D < T_BC] for (AC, GT) — eq. Chat in paper.tex.

        ĉ = g * (2*max(g, h) - h) / (2 * max(g, h)^2)
    with g = Δq_1^GT(A1D), h = Δq_2^AC(BC).
    """
    gamma, eps = _resolve_params(d=d, delta=delta, epsilon=epsilon, e=e)
    g = target_delta_q_GT(t, r, p, s, eps, gamma, state_idx=1)
    h = target_delta_q_AC(t, r, p, s, eps, gamma)
    M = max(g, h)
    if np.isclose(M, 0.0):
        return float("nan")
    return float(g * (2.0 * M - h) / (2.0 * M ** 2))


def theoretical_C_hat_from_ACGT(t=1, r=0.7, p=0.2, s=0, epsilon=None, e=None,
                                 delta=None, d=None):
    """Alias for calculate_C_hat (notebook / sim validation)."""
    return calculate_C_hat(
        t=t, r=r, p=p, s=s,
        epsilon=epsilon if epsilon is not None else e,
        d=d if d is not None else delta,
        delta=delta, e=e,
    )


def calculate_C_hat_mc(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6,
                       n_samples=200_000, seed=0, delta=None, e=None):
    """MC check of ĉ = P[T_A1D < T_BC] under uniform init on [-L,0]^2."""
    gamma, eps = _resolve_params(d=d, delta=delta, epsilon=epsilon, e=e)
    rng = np.random.default_rng(seed)
    L = 1.0 / (1.0 - gamma)
    tgt_a1d = target_delta_q_GT(t, r, p, s, eps, gamma, state_idx=1)
    tgt_bc = target_delta_q_AC(t, r, p, s, eps, gamma)
    hits = 0
    for _ in range(n_samples):
        x = rng.uniform(-L, 0.0)
        y = rng.uniform(-L, 0.0)
        if acgt_a1d_first_indicator(x, y, tgt_a1d, tgt_bc) == 1.0:
            hits += 1
    return hits / n_samples


def theoretical_D_hat_from_ACAD(t=1, r=0.7, p=0.2, s=0, epsilon=None, e=None,
                                 delta=None, d=None, n_mc=200_000):
    """Alias for calculate_D_hat (notebook / sim validation)."""
    return calculate_D_hat(
        t=t, r=r, p=p, s=s,
        epsilon=epsilon if epsilon is not None else e,
        d=d if d is not None else delta,
        delta=delta, e=e, n_mc=n_mc,
    )


def _hitting_log_time(x0, target):
    """Continuous-limit log hitting time for flow toward target from x0."""
    denom = -target
    if np.isclose(denom, 0.0):
        return np.inf
    ratio = (x0 - target) / denom
    if ratio <= 0:
        return np.inf
    return float(np.log(ratio))


def gt_basin_indicator_adgt(x, y, g1_gt, g2_ad):
    """
    1 iff (x, y) in BoA(pi^GT) for (AD, GT) in BC (α→0).

    g1_gt = Δq_1^GT(BC) < 0, g2_ad = Δq_2^AD(BC) > 0; typically x > 0, y < 0.
  GT if player 1 reaches 0 before player 2: T_1 < T_2.
    """
    t1 = _hitting_log_time(x, g1_gt)
    t2 = _hitting_log_time(y, g2_ad)
    if np.isinf(t1) and np.isinf(t2):
        return 0.5
    if np.isinf(t1):
        return 0.0
    if np.isinf(t2):
        return 1.0
    return 1.0 if t1 < t2 else 0.0


def acgt_a1d_first_indicator(x_a1d, y_bc, tgt_a1d, tgt_bc):
    """1 iff T_A1D < T_BC (Ĉ first-crossing region) from (AC, GT) init."""
    t_a = _hitting_log_time(x_a1d, tgt_a1d)
    t_b = _hitting_log_time(y_bc, tgt_bc)
    if np.isinf(t_a) or np.isinf(t_b):
        return np.nan
    return 1.0 if t_a < t_b else 0.0


def _boa_gt_indicator_gtad(x_hat, y_hat, g1_ad, g2_gt):
    """
    Return 1 if (x_hat, y_hat) leads to π^GT from profile (GT, AD) in BC (α→0).

    Entry plane: x_hat ∈ [-L, 0], y_hat ∈ [0, L]; g1_ad > 0, g2_gt < 0.
    GT if T_2(y_hat) < T_1(x_hat) (eq. Dhat-g in paper.tex).
    """
    if g1_ad <= 0 or g2_gt >= 0:
        return 0.5
    if x_hat > 0:
        return 0.0

    t1 = _hitting_log_time(x_hat, g1_ad)
    t2 = _hitting_log_time(y_hat, g2_gt)
    if np.isinf(t1) and np.isinf(t2):
        return 0.5
    if np.isinf(t2):
        return 0.0
    if np.isinf(t1):
        return 1.0
    return 1.0 if t2 < t1 else 0.0


def _propagate_to_xhat_yhat(x1, x2, y1, x_star, y_star):
    """BC position when A1D (player 1) boundary hit first (eq. 1291)."""
    if np.isclose(x_star - x2, 0.0):
        return np.nan, np.nan
    x_hat = x_star + x_star * (x1 - x_star) / (x_star - x2)
    factor = x_star / (x_star - x2)
    y_hat = y_star + factor * (y1 - y_star)
    return x_hat, y_hat


def calculate_D_hat_mc(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6,
                       n_samples=200_000, seed=0, delta=None, e=None):
    """
    Monte Carlo estimate of D̂ = BoA(π^GT | (AC, AD)).

    Only the A1D-first branch contributes (BC-first → (AGT,AD) → BoA 0).
    D̂ = (1/2) * E[g(x̂, ŷ) | |X1| > |X2|].
    """
    gamma, eps = _resolve_params(d=d, delta=delta, epsilon=epsilon, e=e)
    rng = np.random.default_rng(seed)
    L = 1.0 / (1.0 - gamma)

    x_star = target_delta_q_AD(t, r, p, s, eps, gamma)
    y_star = target_delta_q_AC(t, r, p, s, eps, gamma)
    g1_ad = target_delta_q_AD(t, r, p, s, eps, gamma)
    g2_gt = target_delta_q_GT(t, r, p, s, eps, gamma, state_idx=0)

    gt_count = 0.0
    valid = 0

    for _ in range(n_samples):
        x2 = rng.uniform(-L, 0.0)
        x1 = rng.uniform(-L, x2)  # A1D-first: |x1| > |x2|
        y1 = rng.uniform(-L, 0.0)

        x_hat, y_hat = _propagate_to_xhat_yhat(x1, x2, y1, x_star, y_star)
        if np.isnan(x_hat):
            continue
        valid += 1
        gt_count += _boa_gt_indicator_gtad(x_hat, y_hat, g1_ad, g2_gt)

    if valid == 0:
        return float("nan")
    return 0.5 * (gt_count / valid)


def calculate_D_hat(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6,
                    delta=None, e=None, n_mc=200_000):
    """D̂ = BoA(π^GT | (AC, AD)); A1D-first branch only, times Pr[A1D first]=1/2."""
    return calculate_D_hat_mc(t, r, p, s, epsilon, d, n_samples=n_mc, delta=delta, e=e)


def _p_xhat_density(x_hat, x_star, gamma):
    """Marginal p(x_hat | |x1|>|x2|) — NOT a factor in a joint density (x_hat correlates with y_hat via x2)."""
    L = 1.0 / (1.0 - gamma)
    if x_hat <= -L or x_hat >= 0 or np.isclose(x_hat, x_star):
        return 0.0
    t = x_hat + L
    denom = x_hat - x_star
    return 2 * (1 - gamma) ** 2 * x_star * (0.5 * t ** 2 / denom ** 2 - t / denom)


def _p_yhat_density(y_hat, x_star, y_star, gamma):
    """Marginal p(y_hat | |x1|>|x2|), piecewise — NOT independent of x_hat marginally."""
    L = 1.0 / (1.0 - gamma)
    if y_hat < 0 or y_hat > L:
        return 0.0
    y1 = y_star / (1 + x_star * (1 - gamma))
    y2 = (x_star + y_star) / (1 + x_star * (1 - gamma))
    c3 = 2 * (1 - gamma) ** 3 / x_star

    def integral(x2_lo, x2_hi):
        return c3 * ((x2_hi ** 2 * x_star / 2 - x2_hi ** 3 / 3
                       + x_star * x2_hi / (1 - gamma) - x2_hi ** 2 / (2 * (1 - gamma)))
                      - (x2_lo ** 2 * x_star / 2 - x2_lo ** 3 / 3
                         + x_star * x2_lo / (1 - gamma) - x2_lo ** 2 / (2 * (1 - gamma))))

    if 0 <= y_hat <= y1:
        x2_lo = x_star * y_star / (y_hat - y_star) + x_star
        return integral(x2_lo, 0.0)
    if y1 < y_hat <= y2:
        return (6 * x_star * (1 - gamma) - 3 * x_star * (1 - gamma) + 1) / (3 * x_star)
    if y2 < y_hat <= L:
        x2_lo = x_star - (L - y_star) * x_star / (y_hat - y_star)
        return integral(x2_lo, 0.0)
    return 0.0


def calculate_D_hat_factorized_wrong(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6,
                                     delta=None, e=None, n_grid=200):
    """
    Incorrect estimator: ∫∫ g(x,y) p(x) p(y) treating marginals as independent.
    Kept for diagnostics only (~20% bias vs honest MC).
    """
    gamma, eps = _resolve_params(d=d, delta=delta, epsilon=epsilon, e=e)
    L = 1.0 / (1.0 - gamma)
    x_star = target_delta_q_AD(t, r, p, s, eps, gamma)
    y_star = target_delta_q_AC(t, r, p, s, eps, gamma)
    g1_ad = x_star
    g2_gt = target_delta_q_GT(t, r, p, s, eps, gamma, state_idx=0)

    xs = np.linspace(-L + 1e-6, -1e-6, n_grid)
    ys = np.linspace(1e-6, L - 1e-6, n_grid)

    px = np.clip([_p_xhat_density(x, x_star, gamma) for x in xs], 0, None)
    py = np.clip([_p_yhat_density(y, x_star, y_star, gamma) for y in ys], 0, None)
    g_grid = np.array([[_boa_gt_indicator_gtad(x, y, g1_ad, g2_gt) for y in ys] for x in xs])
    integrand = np.outer(px, py) * g_grid
    p_gt_branch = np.trapz(np.trapz(integrand, ys, axis=1), xs)
    return 0.5 * float(p_gt_branch)


def calculate_D_hat_quadrature(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6,
                               delta=None, e=None, n_x2=60, n_inner=40):
    """
    Nested quadrature for eq:Dhat-branch: outer x2, inner (x1, y1) at fixed x2.
    A1D-first branch only; returns same 0.5 weight as legacy helper (GT-branch factor
    applied by caller). Use calculate_D_hat_mc for production.
    """
    gamma, eps = _resolve_params(d=d, delta=delta, epsilon=epsilon, e=e)
    L = 1.0 / (1.0 - gamma)
    x_star = target_delta_q_AD(t, r, p, s, eps, gamma)
    y_star = target_delta_q_AC(t, r, p, s, eps, gamma)
    g1_ad = x_star
    g2_gt = target_delta_q_GT(t, r, p, s, eps, gamma, state_idx=0)

    dens = 2 * (1 - gamma) ** 3
    x2_grid = np.linspace(-L, 0.0, n_x2)
    total = 0.0

    for i in range(len(x2_grid) - 1):
        x2_lo, x2_hi = x2_grid[i], x2_grid[i + 1]
        x2_mid = 0.5 * (x2_lo + x2_hi)
        x1_grid = np.linspace(-L, x2_mid, n_inner)
        y1_grid = np.linspace(-L, 0.0, n_inner)
        g_vals = np.zeros((len(x1_grid), len(y1_grid)))
        for ix, x1 in enumerate(x1_grid):
            for iy, y1 in enumerate(y1_grid):
                x_hat, y_hat = _propagate_to_xhat_yhat(x1, x2_mid, y1, x_star, y_star)
                if not np.isnan(x_hat):
                    g_vals[ix, iy] = _boa_gt_indicator_gtad(x_hat, y_hat, g1_ad, g2_gt)
        inner = np.trapz(np.trapz(g_vals, y1_grid, axis=1), x1_grid)
        total += inner * dens * (x2_hi - x2_lo)

    return 0.5 * float(total)


def diagnose_D_hat_joint(n_samples=100_000, seed=0, **kwargs):
    """
    Compare honest joint MC vs incorrect marginal factorization.
    Returns dict with corr(x_hat, y_hat), D_hat_joint, D_hat_factorized_wrong.
    """
    gamma, eps = _resolve_params(
        d=kwargs.get("d"), delta=kwargs.get("delta"),
        epsilon=kwargs.get("epsilon"), e=kwargs.get("e"),
    )
    t = kwargs.get("t", 1)
    r = kwargs.get("r", 0.7)
    p = kwargs.get("p", 0.2)
    s = kwargs.get("s", 0)
    L = 1.0 / (1.0 - gamma)
    x_star = target_delta_q_AD(t, r, p, s, eps, gamma)
    y_star = target_delta_q_AC(t, r, p, s, eps, gamma)
    g1_ad = x_star
    g2_gt = target_delta_q_GT(t, r, p, s, eps, gamma, state_idx=0)

    rng = np.random.default_rng(seed)
    x_hats, y_hats, g_vals = [], [], []
    for _ in range(n_samples):
        x2 = rng.uniform(-L, 0.0)
        x1 = rng.uniform(-L, x2)
        y1 = rng.uniform(-L, 0.0)
        x_hat, y_hat = _propagate_to_xhat_yhat(x1, x2, y1, x_star, y_star)
        if np.isnan(x_hat):
            continue
        x_hats.append(x_hat)
        y_hats.append(y_hat)
        g_vals.append(_boa_gt_indicator_gtad(x_hat, y_hat, g1_ad, g2_gt))

    x_hats = np.array(x_hats)
    y_hats = np.array(y_hats)
    corr = float(np.corrcoef(x_hats, y_hats)[0, 1]) if len(x_hats) > 1 else float("nan")
    d_joint = float(np.mean(g_vals)) if g_vals else float("nan")
    d_wrong = calculate_D_hat_factorized_wrong(**kwargs)
    return {
        "corr_xhat_yhat_a1d_first": corr,
        "E_g_honest_joint_a1d_branch": d_joint,  # E[g | |x1|>|x2|], not full D_hat
        "factorized_marginal_estimate": d_wrong,  # includes erroneous 0.5 GT-branch prefactor
        "D_hat_mc_full": calculate_D_hat_mc(n_samples=min(n_samples, 50_000), seed=seed, **kwargs),
        "relative_error_factorized_vs_a1d_branch": (d_wrong - d_joint) / d_joint if d_joint else float("nan"),
    }


def get_aggregate_bounds_prop17(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6,
                                 delta=None, e=None):
    """
    Proposition 17 aggregate bounds for BoA(π^GT):
        LB = (1/16)(1 + 2B̂ + 2Ĉ + 2D̂)
        UB = (1/16)(8 + 2B̂ + 2D̂)
    """
    gamma, eps = _resolve_params(d=d, delta=delta, epsilon=epsilon, e=e)
    b_hat = calculate_B_hat(t, r, p, s, eps, gamma)
    c_hat = calculate_C_hat(t, r, p, s, eps, gamma)
    d_hat = calculate_D_hat(t, r, p, s, eps, gamma)
    lower = (1.0 / 16.0) * (1.0 + 2.0 * b_hat + 2.0 * c_hat + 2.0 * d_hat)
    upper = (1.0 / 16.0) * (8.0 + 2.0 * b_hat + 2.0 * d_hat)
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Map paper hats to saved experiment profiles (read existing CSVs, no new sims)
# ---------------------------------------------------------------------------
HAT_PROFILE_MAP = {
    "B_hat": {
        "profile": ("AD", "GT"),
        "hat_fn": "calculate_B_hat",
        "sim_column": "P_GT",
        "csv_0702": "results_vect_paramsfixed_time2000_ADGT.csv",  # rows 225+
        "csv_0805": "results_vect_paramsfixed_time2000_ADGT.csv",  # rows :225
        "slice_0702": slice(225, None),
        "slice_0805": slice(None, 225),
    },
    "C_hat": {
        "profile": ("AC", "GT"),
        "hat_fn": "calculate_C_hat",
        "sim_columns": ("P_direct_GT_analytic", "P_direct_GT"),
        "boa_column": "P_GT",
        "csv_0702": "ACGT_Chat_validate_0702.csv",
        "csv_0805": "ACGT_Chat_validate_0805.csv",
        "fallback_csv_0702": "results_vect_paramsfixed_time2000_ACGT0702.csv",
        "fallback_csv_0805": "results_vect_paramsfixed_time2000_ACGT.csv",
    },
    "D_hat": {
        "profile": ("AC", "AD"),
        "hat_fn": "calculate_D_hat",
        "sim_column": "P_GT",
        "csv_0702": "results_vect_paramsfixed_time2000_ACAD0702.csv",
        "csv_0805": "results_vect_paramsfixed_time2000_ACAD0805.csv",
    },
}

PAYOFF_PRESETS = {
    "0702": dict(t=1, r=0.7, p=0.2, s=0),
    "0805": dict(t=1, r=0.8, p=0.5, s=0),
}


def _default_results_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "results"))


def load_saved_experiment_csv(name, results_dir=None):
    """Load a CSV from results/ (or absolute path)."""
    import os
    results_dir = results_dir or _default_results_dir()
    path = name if os.path.isabs(name) else os.path.join(results_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    import pandas as pd
    return pd.read_csv(path)


def add_hat_column(df, hat_name, t=1, r=0.7, p=0.2, s=0, n_mc=50_000):
    """Add B_hat / C_hat / D_hat column from (delta, e) rows."""
    import pandas as pd
    df = df.copy()
    fn = {"B_hat": calculate_B_hat, "C_hat": calculate_C_hat, "D_hat": calculate_D_hat}[hat_name]
    col = hat_name

    cache = {}
    for d, e in df[["delta", "e"]].drop_duplicates().itertuples(index=False):
        key = (float(d), float(e))
        kwargs = dict(t=t, r=r, p=p, s=s, epsilon=key[1], d=key[0])
        if hat_name == "D_hat":
            kwargs["n_mc"] = n_mc
        cache[key] = fn(**kwargs)

    df[col] = df.apply(lambda row: cache[(float(row["delta"]), float(row["e"]))], axis=1)
    return df


def summarize_hat_vs_saved_csv(
    hat_name,
    preset="0702",
    results_dir=None,
    alpha=0.01,
    lambda_val=0.01,
    n_mc=50_000,
    display_rows=False,
):
    """
    Compare paper hat to existing simulation CSV for the matching profile.

    - B_hat  ↔ (AD,GT)  P_GT
    - C_hat  ↔ (AC,GT)  P_direct_GT_analytic (Chat validate CSVs)
    - D_hat  ↔ (AC,AD)  P_GT
    """
    import os
    import pandas as pd

    meta = HAT_PROFILE_MAP[hat_name]
    pay = PAYOFF_PRESETS[preset]
    t, r, p, s = pay["t"], pay["r"], pay["p"], pay["s"]

    csv_key = f"csv_{preset}"
    csv_name = meta[csv_key]
    df = load_saved_experiment_csv(csv_name, results_dir)
    sl_key = f"slice_{preset}"
    if sl_key in meta:
        df = df.iloc[meta[sl_key]]

    df = add_hat_column(df, hat_name, t=t, r=r, p=p, s=s, n_mc=n_mc)
    hat_col = hat_name if hat_name in df.columns else hat_name  # B_hat etc.

    if alpha is not None:
        df = df[df["alpha"] == alpha]
    if lambda_val is not None:
        df = df[df["lambda"] == lambda_val]

    out = {
        "hat": hat_name,
        "preset": preset,
        "profile": meta["profile"],
        "payoff": pay,
        "n_rows": len(df),
        "csv": csv_name,
    }

    if hat_name == "C_hat":
        for col in meta["sim_columns"]:
            err = df[col] - df[hat_col]
            out[f"mae_{col}"] = float(err.abs().mean())
            out[f"max_err_{col}"] = float(err.abs().max())
        boa_err = df[meta["boa_column"]] - df[hat_col]
        out["mae_P_GT"] = float(boa_err.abs().mean())
        out["note"] = "P_GT is full BoA(GT|AC,GT), expected >= C_hat"
    else:
        col = meta["sim_column"]
        err = df[col] - df[hat_col]
        out["mae"] = float(err.abs().mean())
        out["max_err"] = float(err.abs().max())
        out["sim_column"] = col

    if display_rows:
        cols = ["delta", "e", "alpha", "lambda", hat_col]
        if hat_name == "C_hat":
            cols += list(meta["sim_columns"]) + [meta["boa_column"]]
        else:
            cols.append(meta["sim_column"])
        out["table"] = df[cols].round(4)

    return out


def print_hat_validation_summary(results_dir=None, n_mc=50_000):
    """Print MAE table for B_hat, C_hat, D_hat against all saved CSV presets."""
    for hat in ("B_hat", "C_hat", "D_hat"):
        print(f"\n{'='*60}\n{hat}  profile {HAT_PROFILE_MAP[hat]['profile']}\n{'='*60}")
        for preset in ("0702", "0805"):
            try:
                s = summarize_hat_vs_saved_csv(
                    hat, preset=preset, results_dir=results_dir, n_mc=n_mc,
                )
            except FileNotFoundError as exc:
                print(f"  {preset}: missing {exc}")
                continue
            pay = s["payoff"]
            print(f"  {preset} (R={pay['r']}, P={pay['p']}) — {s['csv']} ({s['n_rows']} rows)")
            if hat == "C_hat":
                print(
                    f"    P_direct_GT_analytic: MAE={s['mae_P_direct_GT_analytic']:.4f}, "
                    f"max={s['max_err_P_direct_GT_analytic']:.4f}"
                )
                print(
                    f"    P_direct_GT (traj):   MAE={s['mae_P_direct_GT']:.4f}, "
                    f"max={s['max_err_P_direct_GT']:.4f}"
                )
            else:
                print(
                    f"    {s['sim_column']}: MAE={s['mae']:.4f}, max={s['max_err']:.4f}"
                )


def hitting_time_probability_C_hat(dq_a1d, dq_bc, target_a1d, target_bc):
    """P[T_A1D < T_BC] from initial Δq values (continuous limit)."""
    t_a = _hitting_log_time(dq_a1d, target_a1d)
    t_b = _hitting_log_time(dq_bc, target_bc)
    if np.isinf(t_a) or np.isinf(t_b):
        return np.nan
    return float(t_a < t_b)


def _shade_mask(ax, X, Y, mask, color, alpha=0.22):
    """Fill boolean mask on axis."""
    ax.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5], colors=[color], alpha=alpha)


def _boa_line_from_targets(ax, g1, g2, gamma, color='red', lw=2, label='BoA boundary'):
    """Plot BoAB segment from origin to axis intercepts (eq. Bhat geometry)."""
    scale = min(1.0 / ((1.0 - gamma) * abs(g1) + 1e-8),
                1.0 / ((1.0 - gamma) * abs(g2) + 1e-8))
    xs = np.array([0.0, -scale * g1])
    ys = np.array([0.0, -scale * g2])
    ax.plot(xs, ys, color=color, linewidth=lw, label=label)
    return xs, ys


def _hitting_boundary_line(xmin, xmax, ymin, ymax, t1, t2):
    """
    Curve (t_a = t_b) in (x, y): (x-t1)/(-t1) = (y-t2)/(-t2) for negative init.
    Returns (xs, ys) clipped to box.
    """
    if np.isclose(t1, 0) or np.isclose(t2, 0):
        return None, None
    xs = np.linspace(xmin, xmax, 200)
    ys = t2 + (t2 / t1) * (xs - t1)
    ok = (ys >= ymin) & (ys <= ymax)
    return xs[ok], ys[ok]


def plot_adgt_boa_validation(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6,
                             q1=None, q2=None, strategy1='AD', strategy2='GT',
                             alpha=0.01, density=18, ax=None, title=None):
    """
    Plot BC dynamics for (AD, GT) with shaded GT basin (B̂) and optional trajectory.
    """
    from MultiAgent_DQ import func_map

    gamma = d
    g1 = target_delta_q_GT(t, r, p, s, epsilon, gamma, state_idx=0)
    g2 = target_delta_q_AD(t, r, p, s, epsilon, gamma)
    L = 1.0 / (1.0 - gamma)
    xmin, xmax, ymin, ymax = 0.0, L, -L, 0.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    X, Y = np.meshgrid(np.linspace(xmin, xmax, density), np.linspace(ymin, ymax, density))
    mask = np.vectorize(lambda x, y: gt_basin_indicator_adgt(x, y, g1, g2) > 0.5)(X, Y)
    _shade_mask(ax, X, Y, mask, color='#2ca02c')
    from matplotlib.lines import Line2D
    shade_legend = Line2D([0], [0], color='#2ca02c', lw=8, alpha=0.35, label=r'BoA$(\pi^{GT})$')

    vf = VectorFieldAutomatic(
        q1=q1, q2=q2, t=t, r=r, p=p, s=s, epsilon=epsilon, d=gamma,
        func1=func_map[strategy2], func2=func_map[strategy1],
        strategy1=strategy1, strategy2=strategy2, alpha=alpha, density=density,
        show_trajectory=q1 is not None, show_basin=False,
    )
    vf.add_vector_field(ax, xmin, xmax, ymin, ymax, vf.func1, vf.func2, 0)
    _boa_line_from_targets(ax, g1, g2, gamma)

    if q1 is not None:
        ax.quiver(vf.xC[:-1], vf.yC[:-1], vf.uC, vf.vC,
                  angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.85,
                  width=0.004, label='Trajectory BC')

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel(r'Agent 1 $\Delta Q$ (BC)')
    ax.set_ylabel(r'Agent 2 $\Delta Q$ (BC)')
    b_hat = calculate_B_hat(t, r, p, s, epsilon, gamma)
    ax.set_title(title or rf'$(AD,GT)$: $\hat{{B}}={b_hat:.3f}$')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [shade_legend], fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)
    return fig, ax


def plot_acgt_boundary_validation(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6,
                                  q1=None, q2=None, alpha=0.01, density=18,
                                  figsize=(12, 5), title=None):
    """
    Joint (Δq_1 A1D, Δq_2 BC) plot with Ĉ first-crossing region and trajectories.
    """
    gamma = d
    tgt_a1d = target_delta_q_GT(t, r, p, s, epsilon, gamma, state_idx=1)
    tgt_bc = target_delta_q_AC(t, r, p, s, epsilon, gamma)
    L = 1.0 / (1.0 - gamma)
    xmin, xmax, ymin, ymax = -L, 0.0, -L, 0.0

    fig, ax = plt.subplots(figsize=figsize)
    X, Y = np.meshgrid(np.linspace(xmin, xmax, density), np.linspace(ymin, ymax, density))
    mask = np.vectorize(
        lambda x, y: acgt_a1d_first_indicator(x, y, tgt_a1d, tgt_bc) == 1.0
    )(X, Y)
    _shade_mask(ax, X, Y, mask, color='#ff7f0e')
    from matplotlib.lines import Line2D
    shade_legend = Line2D([0], [0], color='#ff7f0e', lw=8, alpha=0.35, label=r'$T_{A1D}<T_{BC}$ (Ĉ region)')

    bx, by = _hitting_boundary_line(xmin, xmax, ymin, ymax, tgt_a1d, tgt_bc)
    if bx is not None:
        ax.plot(bx, by, 'r-', lw=2, label='First-crossing boundary')

    if q1 is not None and q2 is not None:
        x_a1d = q1['Q4'].values - q1['Q2'].values
        y_bc = q2['Q3'].values - q2['Q1'].values
        ax.plot(x_a1d, y_bc, 'b-', lw=1.5, alpha=0.9, label='Trajectory')
        ax.scatter(x_a1d[0], y_bc[0], c='blue', s=40, zorder=5, label='Start')
        in_c = acgt_a1d_first_indicator(x_a1d[0], y_bc[0], tgt_a1d, tgt_bc) == 1.0
        ax.scatter(x_a1d[-1], y_bc[-1], c='green' if in_c else 'red', s=40, zorder=5)

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel(r'Agent 1 $\Delta Q$ (A1D)')
    ax.set_ylabel(r'Agent 2 $\Delta Q$ (BC)')
    c_hat = calculate_C_hat(t, r, p, s, epsilon, gamma)
    ax.set_title(title or rf'$(AC,GT)$: $\hat{{C}}={c_hat:.3f}$')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [shade_legend], fontsize=8)
    ax.grid(True, alpha=0.2)
    return fig, ax


def plot_acad_entry_validation(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6,
                               entry_points=None, density=22, figsize=(6, 5), title=None):
    r"""Entry plane (x_hat, y_hat) for (GT,AD) with shaded GT basin (D_hat)."""
    gamma = d
    g1 = target_delta_q_AD(t, r, p, s, epsilon, gamma)
    g2 = target_delta_q_GT(t, r, p, s, epsilon, gamma, state_idx=0)
    L = 1.0 / (1.0 - gamma)
    xmin, xmax, ymin, ymax = -L, 0.0, 0.0, L

    fig, ax = plt.subplots(figsize=figsize)
    X, Y = np.meshgrid(np.linspace(xmin, xmax, density), np.linspace(ymin, ymax, density))
    mask = np.vectorize(lambda x, y: _boa_gt_indicator_gtad(x, y, g1, g2) > 0.5)(X, Y)
    _shade_mask(ax, X, Y, mask, color='#9467bd')
    from matplotlib.lines import Line2D
    shade_legend = Line2D([0], [0], color='#9467bd', lw=8, alpha=0.35, label=r'BoA$(\pi^{GT}|GT,AD)$')

    bx, by = _hitting_boundary_line(xmin, xmax, ymin, ymax, g1, g2)
    if bx is not None:
        ax.plot(bx, by, 'r-', lw=2, label=r'$T_2=T_1$ boundary')

    if entry_points is not None:
        xs, ys = entry_points
        ax.scatter(xs, ys, c='blue', s=25, alpha=0.7, label=r'Sim entry $(\hat{x},\hat{y})$')

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\hat{x}=\Delta q_1(BC)$')
    ax.set_ylabel(r'$\hat{y}=\Delta q_2(BC)$')
    d_hat = calculate_D_hat(t, r, p, s, epsilon, gamma, n_mc=100_000)
    ax.set_title(title or rf'$(AC,AD)\to(GT,AD)$: $\hat{{D}}={d_hat:.3f}$')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [shade_legend], fontsize=8)
    ax.grid(True, alpha=0.2)
    return fig, ax


def verify_trajectory_in_basin(q1, q2, case, t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6):
    """
    Check that each BC/A1D trajectory point lies in the analytic basin for its case.

    Returns dict with fractions in-basin and endpoint strategy guess.
    """
    gamma = d
    if case == 'ADGT':
        g1 = target_delta_q_GT(t, r, p, s, epsilon, gamma, state_idx=0)
        g2 = target_delta_q_AD(t, r, p, s, epsilon, gamma)
        xs = q1['Q3'].values - q1['Q1'].values
        ys = q2['Q3'].values - q2['Q1'].values
        ind = np.array([gt_basin_indicator_adgt(x, y, g1, g2) for x, y in zip(xs, ys)])
        same = bool(np.all(ind == ind[0]))
        return {
            'stays_in_initial_basin': same,
            'initial_basin': 'GT' if ind[0] > 0.5 else 'AD',
            'frac_in_gt_basin': float(np.mean(ind > 0.5)),
            'n': len(xs),
        }
    if case == 'ACGT':
        tgt_a1d = target_delta_q_GT(t, r, p, s, epsilon, gamma, state_idx=1)
        tgt_bc = target_delta_q_AC(t, r, p, s, epsilon, gamma)
        x0 = q1['Q4'].values[0] - q1['Q2'].values[0]
        y0 = q2['Q3'].values[0] - q2['Q1'].values[0]
        ind0 = acgt_a1d_first_indicator(x0, y0, tgt_a1d, tgt_bc)
        return {'initial_in_C_region': float(ind0 == 1.0), 'x0': x0, 'y0': y0}
    raise ValueError(f"Unknown case {case}")


def sample_acad_entry_points(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6, n=200, seed=0):
    """Sample (x_hat, y_hat) on A1D-first path only (eq. propagation)."""
    gamma, eps = d, epsilon
    rng = np.random.default_rng(seed)
    L = 1.0 / (1.0 - gamma)
    x_star = target_delta_q_AD(t, r, p, s, eps, gamma)
    y_star = target_delta_q_AC(t, r, p, s, eps, gamma)
    xs, ys = [], []
    for _ in range(n):
        x2 = rng.uniform(-L, 0.0)
        x1 = rng.uniform(-L, x2)
        y1 = rng.uniform(-L, 0.0)
        xh, yh = _propagate_to_xhat_yhat(x1, x2, y1, x_star, y_star)
        if np.isnan(xh):
            continue
        xs.append(xh)
        ys.append(yh)
    return np.array(xs), np.array(ys)


def delta_q_from_q(q, state_idx):
    """Δq = q[D] - q[C] for one agent matrix q[action, state]."""
    return float(q[1, state_idx] - q[0, state_idx])


def detect_boundary_crossing_acgt(q1_hist, q2_hist):
    """
    Detect which axis is crossed first for (AC, GT) dynamics.

    Returns 'A1D_first' if Δq_1(A1D) crosses zero before Δq_2(BC),
    'BC_first' otherwise, or None if no crossing in history.
    """
    if len(q1_hist) < 2:
        return None

    prev_dq1_a1d = delta_q_from_q(q1_hist[0], 1)
    prev_dq2_bc = delta_q_from_q(q2_hist[0], 0)

    for k in range(1, len(q1_hist)):
        dq1_a1d = delta_q_from_q(q1_hist[k], 1)
        dq2_bc = delta_q_from_q(q2_hist[k], 0)

        crossed_a1d = prev_dq1_a1d < 0 and dq1_a1d >= 0
        crossed_bc = prev_dq2_bc < 0 and dq2_bc >= 0

        if crossed_a1d and crossed_bc:
            return "A1D_first"  # tie-break same as analytic measure-zero set
        if crossed_a1d:
            return "A1D_first"
        if crossed_bc:
            return "BC_first"

        prev_dq1_a1d, prev_dq2_bc = dq1_a1d, dq2_bc

    return None


def analytic_direct_gt_probability_acgt(q1, q2, t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6):
    """Per-sample P[T_A1D < T_BC] from initial Q (α→0 limit)."""
    tgt_a1d = target_delta_q_GT(t, r, p, s, epsilon, d, state_idx=1)
    tgt_bc = target_delta_q_AC(t, r, p, s, epsilon, d)
    dq1_a1d = delta_q_from_q(q1, 1)
    dq2_bc = delta_q_from_q(q2, 0)
    return hitting_time_probability_C_hat(dq1_a1d, dq2_bc, tgt_a1d, tgt_bc)


def calculate_boa_boundary_ADGT(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6, state_idx=0):
    """
    Calculate basin-of-attraction boundary vectors for the (AD, GT) configuration.
    
    This matches the compute_boa method in VectorFieldAutomatic class.
    The boundary is determined by the policy gradient differences scaled by (1-δ).
    
    Formula: 
        scale = min(1/((1-δ)|ΔQx|), 1/((1-δ)|ΔQy|))
        boundary = [(-scale·ΔQx, -scale·ΔQy)]
    
    where ΔQx = Δq_1^GT(BC) and ΔQy = Δq_2^AD(BC) (eq. Bhat).

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
    dx = target_delta_q_GT(t, r, p, s, epsilon, d, state_idx=state_idx)
    dy = target_delta_q_AD(t, r, p, s, epsilon, d)
    
    scale = min(1 / ((1 - d) * abs(dx) + 1e-8),
                1 / ((1 - d) * abs(dy) + 1e-8))
    
    boa_x = np.array([0, -scale * dx])
    boa_y = np.array([0, -scale * dy])
    
    return boa_x, boa_y


def calculate_pi_ADGT(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6, state_idx=0):
    """P(AD) = 1 - B̂ for profile (AD, GT)."""
    return float(1.0 - calculate_B_hat(t, r, p, s, epsilon, d, state_idx=state_idx))


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
def get_aggregate_bounds(t=1, r=0.7, p=0.2, s=0, epsilon=0.1, d=0.6, state_idx=0, P_AD=True):
    """Prop. 17 aggregate bounds; delegates to get_aggregate_bounds_prop17."""
    lower_GT, upper_GT = get_aggregate_bounds_prop17(t, r, p, s, epsilon, d)
    if P_AD:
        return 1.0 - upper_GT, 1.0 - lower_GT
    return lower_GT, upper_GT

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
    P_GTGT_given_ADGT = calculate_B_hat(t, r, p, s, epsilon, d)
    P_GTGT_given_GTAC = 0.5 * (1 + calculate_B_hat(t, r, p, s, epsilon, d))
    P_GTGT_given_ACAD = calculate_D_hat(t, r, p, s, epsilon, d)
    
    lower_bound, upper_bound = get_aggregate_bounds_prop17(t, r, p, s, epsilon, d)
    
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
    print(f"  P(GT,GT | AC,AD)          {P_GTGT_given_ACAD:.4f}")
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
            'P_GTGT_given_ACAD': 0.5 * pi_GTGT,
            'lower_bound': (1/16) * (2 + 4 * pi_GTGT),
            'upper_bound': (1/16) * (7 + 4 * pi_GTGT)
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
    print(f"{'  |ΔQx| (AD gradient)       ' + format(results[0]['abs_dx'], '.3f'):<{w}} |   |ΔQx| (AD gradient)       {results[1]['abs_dx']:.3f}")
    print(f"{'  |ΔQy| (GT gradient)       ' + format(results[0]['abs_dy'], '.3f'):<{w}} |   |ΔQy| (GT gradient)       {results[1]['abs_dy']:.3f}\n")
    
    # Basin analysis
    print(f"{'Basin analysis:             ' + results[0]['basin_analysis']:<{w}} | Basin analysis:             {results[1]['basin_analysis']}")
    print(f"{'  π_small                   ' + format(results[0]['pi_small'], '.3f'):<{w}} |   π_small                   {results[1]['pi_small']:.3f}")
    pad0 = ' ' * (11 - len(results[0]['adad_label']))
    pad1 = ' ' * (11 - len(results[1]['adad_label']))
    print(f"{'  π_ADAD ' + results[0]['adad_label'] + pad0 + format(results[0]['pi_ADAD'], '.3f'):<{w}} |   π_ADAD {results[1]['adad_label']}{pad1}{results[1]['pi_ADAD']:.3f}")
    
    left_gt = f"  π_GTGT {results[0]['gtgt_label']}{' ' * (11 - len(results[0]['gtgt_label']))}{results[0]['pi_GTGT']:.3f}"
    if "(stronger)" in results[0]['gtgt_label']:
        left_gt += "  ★"
    right_gt = f"  π_GTGT {results[1]['gtgt_label']}{' ' * (11 - len(results[1]['gtgt_label']))}{results[1]['pi_GTGT']:.3f}"
    if "(stronger)" in results[1]['gtgt_label']:
        right_gt += "  ★"
    print(f"{left_gt:<{w}} | {right_gt}\n")
    
    # Conditional probabilities
    print(f"{'Conditional probabilities:':<{w}} | Conditional probabilities:")
    print(f"{'  P(GT,GT | GT,GT)          ' + format(results[0]['P_GTGT_given_GTGT'], '.4f'):<{w}} |   P(GT,GT | GT,GT)          {results[1]['P_GTGT_given_GTGT']:.4f}")
    print(f"{'  P(GT,GT | AD,GT)          ' + format(results[0]['P_GTGT_given_ADGT'], '.4f'):<{w}} |   P(GT,GT | AD,GT)          {results[1]['P_GTGT_given_ADGT']:.4f}")
    print(f"{'  P(GT,GT | GT,AC)          ' + format(results[0]['P_GTGT_given_GTAC'], '.4f'):<{w}} |   P(GT,GT | GT,AC)          {results[1]['P_GTGT_given_GTAC']:.4f}")
    print(f"{'  P(GT,GT | AC,AD)          ' + format(results[0]['P_GTGT_given_ACAD'], '.4f'):<{w}} |   P(GT,GT | AC,AD)          {results[1]['P_GTGT_given_ACAD']:.4f}\n")
    
    # Aggregate bounds
    print(f"{'Aggregate bounds:':<{w}} | Aggregate bounds:")
    print(f"{'  Lower bound               ' + format(results[0]['lower_bound'], '.3f'):<{w}} |   Lower bound               {results[1]['lower_bound']:.3f}")
    print(f"{'  Upper bound               ' + format(results[0]['upper_bound'], '.3f'):<{w}} |   Upper bound               {results[1]['upper_bound']:.3f}\n")
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
