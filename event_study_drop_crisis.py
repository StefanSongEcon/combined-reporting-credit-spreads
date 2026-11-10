"""
event_study_drop_crisis.py -- DML DR-DiD event study on the main stack after dropping the
2007-09 enactment cohorts (MI, WV 2007; MA 2008; WI 2009), as in the robustness row
"drop 2007-09 cohorts (crisis)". Same estimator settings as the main event study.

Outputs: results/event_study_drop0709.csv, results/event_study_drop0709.png
         (the figure overlays the main event study for comparison)
Run:     OMP_NUM_THREADS=4 python3 -u event_study_drop_crisis.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import dml_cr as dc
import run_analysis as ra


def main():
    st = pd.read_csv(ra.RES / "stack_main.csv.gz")
    st = st[~st.cohort.between(2007, 2009)].reset_index(drop=True)
    cc = st.groupby(["cohort", "D"]).firm.nunique().unstack()
    print("cohort counts after dropping 2007-09:\n" + cc.to_string(), flush=True)
    es = dc.event_study(st, ra.X_RAW, ks=range(-5, 5), base_k=-1, learner="auto", n_rep=ra.N_REP)
    es.to_csv(ra.RES / "event_study_drop0709.csv", index=False)
    print(es.round(4).to_string(index=False))

    main_es = pd.read_csv(ra.RES / "event_study.csv")
    fig, ax = plt.subplots(figsize=(7, 3.9))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(-0.5, color="0.6", lw=0.8, ls="--")
    off = 0.12
    ax.errorbar(main_es.k - off, main_es.theta, yerr=1.645 * main_es.se, fmt="o", color="#9aa5b1",
                capsize=3, label="all cohorts (main)")
    ax.errorbar(es.k + off, es.theta, yerr=1.645 * es.se, fmt="o", color="#1f4e79", capsize=3,
                label="drop 2007-09 cohorts (MI, WV, MA, WI)")
    ax.set_xlabel("Years relative to enactment of combined reporting")
    ax.set_ylabel("ATT on log credit spread")
    ax.set_title("DML DR-DiD event study, dropping 2007-09 cohorts (90% CI)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(ra.RES / "event_study_drop0709.png", dpi=180)


if __name__ == "__main__":
    main()
