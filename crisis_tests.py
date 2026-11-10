"""
crisis_tests.py -- separate the combined-reporting effect from the 2008-09 crisis.

The TX 2006 cohort is most of the treated sample, and its event years k = 2-3 are 2008-09.
Three tests, each on the saved main stack (results/stack_main.csv.gz):

  A. drop TX     : remove the 2006 (TX) cohort. The remaining cohorts hit 2008-09 at
                   different event times (VT k=4-5, MI/WV k=1-2, MA k=0-1, WI k=-1-0),
                   so a crisis artefact would not line up at one k.
  B. post-2010   : cohorts 2011, 2014, 2015, 2018 only (DC, NY, RI, CT, KY, NJ). Post
                   windows are entirely after the crisis. Few treated units.
  C. crisis ctrl : all cohorts, but let spread trends depend on each firm's pre-period
                   spread level (DR-DiD: pre-period mean log spread added to X; OLS: FE for
                   cohort x year x pre-spread quintile and cohort x year x number-of-states
                   bin). Riskier and more geographically spread-out firms were hit harder
                   in 2008-09; this absorbs that.

Outputs: results/crisis_tests.csv (pooled ATTs), results/crisis_event_studies.csv,
         results/crisis_event_studies.png
Run:     OMP_NUM_THREADS=4 python3 -u crisis_tests.py
"""
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dml_cr as dc
import run_analysis as ra

T0 = time.time()
KS = range(-5, 5)


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def event_study(st, pre_y=False):
    xc = ra.XC + (["pre_y"] if pre_y else [])
    rows = []
    for k in KS:
        if k == -1:
            rows.append((k, 0.0, 0.0, np.nan, np.nan))
            continue
        ld = dc.long_differences(st, ra.X_RAW, k=k, base_k=-1, include_pre_y=pre_y)
        r = dc.dr_did_att(ld, xc, learner="auto", n_rep=ra.N_REP)
        rows.append((k, r["theta"], r["se"], r["n_units"], r["n_treated"]))
    return pd.DataFrame(rows, columns=["k", "theta", "se", "n_units", "n_treated"])


def fe_ols_crisis(st):
    """Paper's stacked FE OLS, with the industry-year-cohort FE replaced by
    cohort x year x pre-spread quintile and cohort x year x n_states bin (two-way demeaning
    on firm-cohort and the combined cell)."""
    d = st.dropna(subset=["y"]).copy()
    pre = d[d.post == 0].groupby("unit").agg(py=("y", "mean"), ns=("n_states", "mean"))
    pre["q_y"] = pd.qcut(pre.py, 5, labels=False, duplicates="drop")
    pre["q_ns"] = pd.qcut(pre.ns.rank(method="first"), 3, labels=False)
    d = d.join(pre[["q_y", "q_ns"]], on="unit").dropna(subset=["q_y", "q_ns"])
    d["Dpost"] = d.D * d.post
    d["fc"] = d.unit
    d["cell"] = (d.cohort.astype(str) + "_" + d.year.astype(str) + "_" + d.q_y.astype(int).astype(str)
                 + "_" + d.q_ns.astype(int).astype(str))
    Z = dc._demean_2way(d, ["y", "Dpost"], "fc", "cell")
    Y, X = Z[:, 0], Z[:, 1:]
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    e = Y - X @ beta
    XtXi = np.linalg.pinv(X.T @ X)
    sc = pd.DataFrame(X * e[:, None]).groupby(d.firm.to_numpy()).sum().to_numpy()
    se = float(np.sqrt((XtXi @ (sc.T @ sc) @ XtXi)[0, 0]))
    return {"theta": float(beta[0]), "se": se, "n_units": int(d.unit.nunique())}


def main():
    st = pd.read_csv(ra.RES / "stack_main.csv.gz")
    specs = {
        "A. drop TX (2006 cohort)": (st[st.cohort != 2006], False),
        "B. post-2010 cohorts only": (st[st.cohort >= 2010], False),
        "C. all cohorts + pre-spread control": (st, True),
    }
    pooled, es_all = [], []
    for name, (s, pre_y) in specs.items():
        s = s.reset_index(drop=True)
        cc = s.groupby(["cohort", "D"]).firm.nunique().unstack(fill_value=0)
        log(f"== {name} ==\ncohort counts (0 = controls, 1 = treated):\n{cc.to_string()}")
        ld = dc.long_differences(s, ra.X_RAW, include_pre_y=pre_y)
        xc = ra.XC + (["pre_y"] if pre_y else [])
        r = dc.dr_did_att(ld, xc, learner="auto", n_rep=ra.N_REP_MAIN)
        o = dc.stacked_fe_ols(s, "Dpost")
        rows = [(name, "DML DR-DiD (auto)", r["theta"], r["se"], r["n_units"], r["n_treated"], r.get("learners")),
                (name, "Stacked FE OLS (paper eq.5)", o["theta"], o["se"], s.unit.nunique(), int(ld.D.sum()), "")]
        if pre_y:
            oc = fe_ols_crisis(s)
            rows.append((name, "Stacked FE OLS, cohort x year x pre-spread x n_states FE",
                         oc["theta"], oc["se"], oc["n_units"], int(ld.D.sum()), ""))
        for row in rows:
            log(f"  {row[1]:58s} theta {row[2]:+.4f}  se {row[3]:.4f}  t {row[2] / row[3]:+.2f}  units {row[4]}")
        pooled += rows
        es = event_study(s, pre_y=pre_y)
        es.insert(0, "spec", name)
        log("event study:\n" + es.round(4).to_string(index=False))
        es_all.append(es)

    pd.DataFrame(pooled, columns=["spec", "estimator", "theta", "se", "n_units", "n_treated", "learners"]) \
        .assign(t=lambda d: d.theta / d.se).to_csv(ra.RES / "crisis_tests.csv", index=False)
    es_all = pd.concat(es_all)
    es_all.to_csv(ra.RES / "crisis_event_studies.csv", index=False)

    main_es = pd.read_csv(ra.RES / "event_study.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, (name, g) in zip(axes, es_all.groupby("spec", sort=False)):
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(-0.5, color="0.6", lw=0.8, ls="--")
        ax.errorbar(main_es.k - 0.12, main_es.theta, yerr=1.96 * main_es.se, fmt="o", color="#9aa5b1",
                    capsize=3, ms=4, label="main (all cohorts)")
        ax.errorbar(g.k + 0.12, g.theta, yerr=1.96 * g.se, fmt="o", color="#1f4e79", capsize=3, label=name[3:])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Years relative to enactment")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    axes[0].set_ylabel("ATT on log credit spread")
    fig.suptitle("Separating combined reporting from the 2008-09 crisis (DML DR-DiD, 95% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ra.RES / "crisis_event_studies.png", dpi=180)
    log("done")


if __name__ == "__main__":
    main()
