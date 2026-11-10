"""
placebo.py -- permutation test: shuffle adoption years across the adopting states and
re-estimate the main DR-DiD ATT. Asks whether "firms exposed to these states vs. clean
controls" shows spread changes of this size around arbitrary years.

Output: results/placebo.csv, results/placebo.png
Run:    OMP_NUM_THREADS=4 python3 -u placebo.py [n_draws]
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dml_cr as dc
import run_analysis as ra

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def main():
    t0 = time.time()
    panel, expo, expo_f, policy = ra.load()
    ev = dc.load_events(ra.HERE / "cr_events.csv", timing="enacted", drop_states=["NM"])
    actual = pd.read_csv(ra.RES / "estimates.csv")
    actual = actual[(actual.block == "main") & (actual.estimator == "DML DR-DiD (rf)")].theta.iloc[0]
    rng = np.random.default_rng(42)
    rows = []
    for b in range(N):
        e2 = ev.copy()
        e2["cohort"] = rng.permutation(ev.cohort.to_numpy())
        if (e2.cohort == ev.cohort).all():
            continue
        try:
            st = dc.build_stack_frozen(panel, expo, policy, e2, verbose=False)
            ld = dc.long_differences(st, ra.X_RAW)
            r = dc.dr_did_att(ld, ra.XC, learner="rf", n_rep=1, seed=b)
            o = dc.ols_long_diff(ld, ra.XC)
            rows.append((b, r["theta"], r["se"], o["theta"], len(ld), int(ld.D.sum())))
        except Exception as ex:  # a draw can leave no treated or no controls
            rows.append((b, np.nan, np.nan, np.nan, 0, 0))
        if (b + 1) % 10 == 0:
            print(f"  {b + 1}/{N} draws ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows, columns=["draw", "theta_dml", "se_dml", "theta_ols", "n_units", "n_treated"])
    df.to_csv(ra.RES / "placebo.csv", index=False)
    v = df.theta_dml.dropna()
    p_two = np.mean(np.abs(v) >= abs(actual))
    p_one = np.mean(v >= actual)
    print(f"actual DML (rf) ATT {actual:.4f}; placebo mean {v.mean():.4f} sd {v.std():.4f}; "
          f"n={len(v)}; p(|placebo|>=|actual|)={p_two:.3f}; p(placebo>=actual)={p_one:.3f}; "
          f"{time.time() - t0:.0f}s")
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.hist(v, bins=25, color="#7f7f7f", alpha=0.8)
    ax.axvline(actual, color="#c00000", lw=2, label=f"actual = {actual:.3f}")
    ax.set_xlabel("ATT with adoption years shuffled across states")
    ax.set_ylabel("draws"); ax.legend(); ax.set_title("Placebo distribution (DML DR-DiD, rf)")
    fig.tight_layout(); fig.savefig(ra.RES / "placebo.png", dpi=180)


if __name__ == "__main__":
    main()
