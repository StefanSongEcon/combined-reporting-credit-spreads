"""
headline_figure.py -- README headline figure: DML DR-DiD event study excluding the Texas 2006
cohort, whose law also replaced the franchise tax with the margin tax. Reads the estimates saved
by crisis_tests.py; no re-estimation.

Output: results/event_study_headline.png
Run:    python3 headline_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES = Path(__file__).resolve().parent / "results"


def main():
    es = pd.read_csv(RES / "crisis_event_studies.csv")
    es = es[es.spec.str.startswith("A.")]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(-0.5, color="0.6", lw=0.8, ls="--")
    ax.errorbar(es.k, es.theta, yerr=1.645 * es.se, fmt="o", color="#1f4e79", capsize=3)
    ax.set_xlabel("Years relative to enactment of combined reporting")
    ax.set_ylabel("ATT on log credit spread")
    ax.set_title("DML DR-DiD event study, excluding Texas (base year k = -1, 90% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RES / "event_study_headline.png", dpi=180)


if __name__ == "__main__":
    main()
