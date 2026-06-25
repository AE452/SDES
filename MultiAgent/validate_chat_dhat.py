#!/usr/bin/env python3
"""Compare paper B_hat, C_hat, D_hat to existing simulation CSVs (no new runs).

See Prop17_hats_reference.md for proofs, mapping, and when sim matches theory.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from basin import HAT_PROFILE_MAP, print_hat_validation_summary


def main():
    results_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "results")
    )
    print("Hat ↔ profile ↔ saved CSV mapping:")
    for hat, meta in HAT_PROFILE_MAP.items():
        print(f"  {hat}: profile {meta['profile']}")
    print(f"\nResults directory: {results_dir}\n")
    print_hat_validation_summary(results_dir=results_dir, n_mc=50_000)


if __name__ == "__main__":
    main()
