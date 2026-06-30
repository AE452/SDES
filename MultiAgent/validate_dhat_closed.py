#!/usr/bin/env python3
"""Numerical validation of the closed-form hats in basin.py (no new simulations).

Phase 2/3 (D̂):  calculate_D_hat_closed  vs  vectorised MC  vs  quadrature,
                 and check it does NOT coincide with the factorized-wrong proxy.
Phase 4   (Ĉ):  calculate_C_hat (eq:Chat)  vs  MC, plus the paper limit
                 (γ→1 then ε→0)  Ĉ → (3P-1)/(3P-R)  when 3P+R>2.

Run:  python MultiAgent/validate_dhat_closed.py
"""
import os
import sys
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import basin as B

PRESETS = [("0702", 1, 0.7, 0.2, 0), ("0805", 1, 0.8, 0.5, 0)]
EPS = (0.05, 0.1, 0.2)
GAM = (0.5, 0.75, 0.95)


def d_hat_mc_ground_truth(t, r, p, s, eps, g, n=3_000_000, seed=11):
    """Uniform-on-triangle MC (rejection), mirrors _boa_gt_indicator_gtad."""
    rng = np.random.default_rng(seed)
    L = 1.0 / (1.0 - g)
    xs = B.target_delta_q_AD(t, r, p, s, eps, g)
    ys = B.target_delta_q_AC(t, r, p, s, eps, g)
    g2 = B.target_delta_q_GT(t, r, p, s, eps, g, state_idx=0)
    if xs <= 0 or g2 >= 0:
        return 0.0
    # Player 1 cooperates (X1,X2 ~ U[-L,0]); player 2 DEFECTS in BC (Y1 ~ U[0,L]).
    X1 = rng.uniform(-L, 0, n)
    X2 = rng.uniform(-L, 0, n)
    Y1 = rng.uniform(0.0, L, n)
    m = X1 < X2
    x1, x2, y1 = X1[m], X2[m], Y1[m]
    mm = xs / (xs - x2)
    xh, yh = xs * (1 - mm) + mm * x1, ys * (1 - mm) + mm * y1
    t1 = np.log((xh - xs) / (-xs))
    ratio = (yh - g2) / (-g2)
    v = ratio > 0
    t2 = np.where(v, np.log(np.where(v, ratio, 1.0)), np.inf)
    return 0.5 * float(((t2 < t1) & v).mean())


def c_hat_mc(t, r, p, s, eps, g, n=3_000_000, seed=13):
    rng = np.random.default_rng(seed)
    L = 1.0 / (1.0 - g)
    ta = B.target_delta_q_GT(t, r, p, s, eps, g, state_idx=1)
    tb = B.target_delta_q_AC(t, r, p, s, eps, g)
    x, y = rng.uniform(-L, 0, n), rng.uniform(-L, 0, n)
    return float((np.log((x - ta) / (-ta)) < np.log((y - tb) / (-tb))).mean())


def main():
    print("=" * 78)
    print("D̂  closed vs MC vs quadrature  (and vs factorized-wrong proxy)")
    print("=" * 78)
    print(f"{'preset':>6} {'eps':>5} {'gam':>5} | {'closed':>8} {'mc':>8} {'quad':>8} "
          f"{'factW':>8} | {'|c-mc|':>7} {'|c-q|':>7} {'|c-fW|':>7}")
    max_cmc = max_cq = 0.0
    min_cfw_nondeg = 9.0
    for name, t, r, p, s in PRESETS:
        for eps in EPS:
            for g in GAM:
                c = B.calculate_D_hat_closed(t, r, p, s, eps, g)
                mc = d_hat_mc_ground_truth(t, r, p, s, eps, g)
                q = B.calculate_D_hat_quadrature(t, r, p, s, eps, g, n_x2=60, n_inner=60)
                fw = B.calculate_D_hat_factorized_wrong(t, r, p, s, eps, g, n_grid=120)
                g2 = B.target_delta_q_GT(t, r, p, s, eps, g, state_idx=0)
                degenerate = g2 >= 0 or B.target_delta_q_AD(t, r, p, s, eps, g) <= 0
                dmc, dq, dfw = abs(c - mc), abs(c - q), abs(c - fw)
                max_cmc, max_cq = max(max_cmc, dmc), max(max_cq, dq)
                if not degenerate:
                    min_cfw_nondeg = min(min_cfw_nondeg, dfw)
                tag = "  (degenerate g2>=0)" if degenerate else ""
                print(f"{name:>6} {eps:>5} {g:>5} | {c:8.4f} {mc:8.4f} {q:8.4f} "
                      f"{fw:8.4f} | {dmc:7.4f} {dq:7.4f} {dfw:7.4f}{tag}")
    print(f"\nMAX |closed-MC|        = {max_cmc:.5f}   (target < 1e-3: {max_cmc < 1e-3})")
    print(f"MAX |closed-quad|      = {max_cq:.5f}   (target < 1e-3: {max_cq < 1e-3})")
    print(f"MIN |closed-factWrong| (non-degenerate) = {min_cfw_nondeg:.5f}   "
          f"(must be >> 0: {min_cfw_nondeg > 1e-2})")

    print("\n" + "=" * 78)
    print("Ĉ  closed (eq:Chat) vs MC")
    print("=" * 78)
    print(f"{'preset':>6} {'eps':>5} {'gam':>5} | {'closed':>8} {'mc':>8} | {'|c-mc|':>7}")
    max_chat = 0.0
    for name, t, r, p, s in PRESETS:
        for eps in EPS:
            for g in GAM:
                c = B.calculate_C_hat(t, r, p, s, eps, g)
                mc = c_hat_mc(t, r, p, s, eps, g)
                d = abs(c - mc)
                max_chat = max(max_chat, d)
                print(f"{name:>6} {eps:>5} {g:>5} | {c:8.4f} {mc:8.4f} | {d:7.4f}")
    print(f"\nMAX |C_hat-MC| = {max_chat:.5f}   (target < 1e-3: {max_chat < 1e-3})")

    print("\n" + "=" * 78)
    print("Ĉ  paper limit  (γ→1 then ε→0):  Ĉ → (3P-1)/(3P-R)  when 3P+R>2")
    print("=" * 78)
    gam = 1 - 1e-10
    for name, t, r, p, s in PRESETS:
        c = B.calculate_C_hat(t, r, p, s, 1e-5, gam)
        valid = (3 * p - r) > 0
        claim = (3 * p - 1) / (3 * p - r) if (3 * p + r > 2) else (3 * p - r) / (4 * (1 - r))
        print(f"  {name}: 3P+R={3*p+r:.2f} 3P-R={3*p-r:+.2f} (regime valid: {valid}) | "
              f"C_hat(lim)={c:.5f}  paper={claim:.5f}")


if __name__ == "__main__":
    main()
