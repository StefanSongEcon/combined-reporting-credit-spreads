"""
run_analysis.py -- full DML analysis: combined reporting -> firm credit spreads.

Inputs (built by the other scripts in this folder)
  firm_panel.csv     gvkey-year spreads + Compustat controls     (build_firm_panel.py)
  exposure_10k.csv   gvkey-fyear-state 10-K exposure shares      (edgar_state_exposure.py)
  state_panel.csv    state-year tax rate, unemployment, GDP      (build_state_panel.py)
  cr_events.csv      adoption dates (Welsch 2025, App. D)

Outputs: results/*.csv, results/event_study.png, results/run_log.txt
Run:     OMP_NUM_THREADS=4 python3 -u run_analysis.py
"""
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dml_cr as dc

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("RESULTS_DIR", HERE / "results"))
RES.mkdir(exist_ok=True)
N_REP_MAIN, N_REP = int(os.environ.get("N_REP_MAIN", 10)), int(os.environ.get("N_REP", 5))
T0 = time.time()
ROWS = []

FIRM_X = ["size", "book_lev", "mkt_lev", "mtb", "roa", "cash_at", "rd_at", "capx_at", "tang",
          "loss", "sales_growth", "age", "roa_vol", "maturity", "log_n_bonds", "log_volume"]
STATE_X = ["nw_tax_rate", "nw_unemp", "nw_gdp_growth", "nw_pre1990_cr", "n_states"]
X_RAW = FIRM_X + STATE_X


def log(msg):
    line = f"[{time.time() - T0:7.0f}s] {msg}"
    print(line, flush=True)
    with open(RES / "run_log.txt", "a") as fh:
        fh.write(line + "\n")


def record(block, spec, est, r, extra=None):
    row = {"block": block, "spec": spec, "estimator": est, "theta": r["theta"], "se": r["se"],
           "t": r["theta"] / r["se"] if r["se"] else np.nan, "n_units": r.get("n_units"),
           "n_treated": r.get("n_treated"), "n_firms": r.get("n_firms")}
    for k in ("auc_m", "r2_l0", "share_m_at_bounds", "r2_l", "r2_m", "learners"):
        if k in r:
            row[k] = r[k]
    if extra:
        row.update(extra)
    ROWS.append(row)
    pd.DataFrame(ROWS).to_csv(RES / "estimates.csv", index=False)
    log(f"{block:<14s} {spec:<34s} {est:<22s} theta={r['theta']: .4f} se={r['se']:.4f} "
        f"t={row['t']: .2f} n={r.get('n_units')}")


# ---------------------------------------------------------------------------
def load():
    panel = pd.read_csv(HERE / "firm_panel.csv", dtype={"firm": str})
    expo = pd.read_csv(os.environ.get("EXPOSURE", HERE / "exposure_10k.csv"), dtype={"firm": str})
    expo = expo[["firm", "year", "state", "share", "source"]]
    policy = pd.read_csv(HERE / "state_panel.csv")
    policy["unemp"] = policy.unemp.fillna(policy.groupby("year").unemp.transform("mean"))
    policy["gdp_growth"] = policy.gdp_growth.fillna(policy.groupby("year").gdp_growth.transform("mean"))

    # carry exposure across years with no usable 10-K (up to 2 years each way) for the
    # year-by-year controls; the frozen design only uses observed pre-period years
    fy = expo[["firm", "year"]].drop_duplicates()
    grid = (fy.groupby("firm").year.agg(["min", "max"]).reset_index())
    grid = pd.concat([pd.DataFrame({"firm": r.firm, "year": range(max(r["min"] - 2, 1998), min(r["max"] + 2, 2025) + 1)})
                      for _, r in grid.iterrows()])
    have = set(map(tuple, fy.to_numpy()))
    src = {}
    years_by_firm = fy.groupby("firm").year.apply(sorted).to_dict()
    for f, ys in years_by_firm.items():
        arr = np.array(ys)
        for t in grid[grid.firm == f].year:
            if (f, t) in have:
                continue
            j = arr[np.argmin(np.abs(arr - t))]
            if abs(j - t) <= 2:
                src[(f, t)] = j
    fill = pd.DataFrame([(f, t, j) for (f, t), j in src.items()], columns=["firm", "year", "from"])
    filled = fill.merge(expo.rename(columns={"year": "from"}), on=["firm", "from"]).drop(columns="from")
    expo_f = pd.concat([expo, filled], ignore_index=True)

    nw = dc.nexus_weight(expo_f, policy.assign(pre1990_cr=policy.pre1990_cr.astype(float)),
                         ["tax_rate", "unemp", "gdp_growth", "pre1990_cr"])
    ns = expo_f[expo_f.share > 0].groupby(["firm", "year"]).state.nunique().rename("n_states").reset_index()
    panel = panel.merge(nw, on=["firm", "year"], how="left").merge(ns, on=["firm", "year"], how="left")
    log(f"panel {len(panel):,} firm-years; with exposure {panel.nw_tax_rate.notna().sum():,}; "
        f"exposure firm-years {expo[['firm','year']].drop_duplicates().shape[0]:,} (+{len(fill):,} filled)")
    srcs = expo.groupby(["firm", "year"]).source.first().value_counts(normalize=True).round(3)
    log("exposure source shares: " + srcs.to_dict().__repr__())
    return panel, expo, expo_f, policy


def make_ld(st, y="y", include_pre_y=True):
    ld = dc.long_differences(st, X_RAW, y=y, include_pre_y=include_pre_y)
    ld["pre_spread_hi"] = (ld["pre_y"] > ld["pre_y"].median()).astype(int) if include_pre_y else 0
    return ld


XC = [f"pre_{c}" for c in X_RAW]


def did_block(block, spec, st, y="y", learners=("auto",), n_rep=N_REP, pre_y_in_x=False,
              dose=True, ols=True):
    ld = make_ld(st, y=y)
    xc = XC + (["pre_y"] if pre_y_in_x else [])
    out = {}
    if ols:
        record(block, spec, "OLS long-diff, no X", dc.ols_long_diff(ld, [], treat="D"))
        record(block, spec, "OLS long-diff, X", dc.ols_long_diff(ld, xc, treat="D"))
    for lr in learners:
        r = dc.dr_did_att(ld, xc, learner=lr, n_rep=n_rep, return_phi=True)
        record(block, spec, f"DML DR-DiD ({lr})", r)
        out[lr] = r
    if dose:
        r = dc.plr(ld, xc, treat="dT", learner=learners[0], n_rep=n_rep)
        record(block, spec, f"DML PLR dose T ({learners[0]})", r)
        r = dc.plr(ld, xc, treat="dA", learner=learners[0], n_rep=n_rep)
        record(block, spec, f"DML PLR geo dose A ({learners[0]})", r)
    return ld, out


# ---------------------------------------------------------------------------
def main():
    open(RES / "run_log.txt", "w").close()
    panel, expo, expo_f, policy = load()
    ev_enact = dc.load_events(HERE / "cr_events.csv", timing="enacted", drop_states=["NM"])
    ev_eff = dc.load_events(HERE / "cr_events.csv", timing="effective", drop_states=["NM"])

    def stack(events=ev_enact, p=panel, **kw):
        return dc.build_stack_frozen(p, expo, policy, events, pre=5, post=4, **kw)

    # ---------------- main specification ----------------
    log("== MAIN: enactment timing, frozen pre-period 10-K exposure, log spread ==")
    st = stack()
    st.to_csv(RES / "stack_main.csv.gz", index=False)
    cc = st.groupby(["cohort", "D"]).firm.nunique().unstack().rename(columns={0: "controls", 1: "treated"})
    cc.to_csv(RES / "cohort_counts.csv")
    log("cohort counts:\n" + cc.to_string())

    ld, out = did_block("main", "baseline", st, learners=("auto", "gbm", "rf", "lasso", "logit"), n_rep=N_REP_MAIN)
    r = out["auto"]
    m = r["m_hat"]
    ov = pd.DataFrame({"D": ld.D, "m_hat": m})
    ov.to_csv(RES / "propensity_main.csv", index=False)
    log("propensity quantiles by group:\n" + ov.groupby("D").m_hat.describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bins = np.linspace(0, 1, 26)
    ax.hist(m[ld.D == 0], bins=bins, alpha=0.6, label="clean controls", color="#7f7f7f")
    ax.hist(m[ld.D == 1], bins=bins, alpha=0.6, label="treated", color="#1f4e79")
    ax.set_xlabel("cross-fitted propensity score"); ax.set_ylabel("firm x cohort units")
    ax.set_title("Overlap, main specification"); ax.legend()
    fig.tight_layout(); fig.savefig(RES / "overlap.png", dpi=180); plt.close(fig)
    bal = dc.balance_table(ld, XC, m)
    bal.to_csv(RES / "balance_main.csv", index=False)
    log("balance:\n" + bal.round(3).to_string(index=False))
    log(f"true-zero check: mean dT among treated = {ld.loc[ld.D == 1, 'dT'].mean():.3f} pp; "
        f"mean dA = {ld.loc[ld.D == 1, 'dA'].mean():.3f}")

    # paper's OLS stacked FE (eq. 5 and eq. 7)
    stc = st.copy()
    record("main", "baseline", "Stacked FE OLS (paper eq.5)", dc.stacked_fe_ols(stc, "Dpost"))
    record("main", "baseline", "Stacked FE OLS + controls", dc.stacked_fe_ols(stc.dropna(subset=X_RAW), "Dpost", X_RAW))
    record("main", "baseline", "Stacked FE OLS dose T (eq.7)", dc.stacked_fe_ols(stc, "T"))

    # overlap-trimmed re-estimation
    keep = (m >= 0.05) & (m <= 0.95)
    ldt = ld[keep].reset_index(drop=True)
    record("main", "overlap 0.05-0.95", "DML DR-DiD (auto)", dc.dr_did_att(ldt, XC, learner="auto", n_rep=N_REP))
    record("main", "overlap 0.05-0.95", "OLS long-diff, X", dc.ols_long_diff(ldt, XC))

    # cohort-specific ATTs
    for e, g in ld.groupby("cohort"):
        if g.D.sum() >= 30 and (1 - g.D).sum() >= 30:
            record("by_cohort", f"cohort {e}", "DML DR-DiD (auto)",
                   dc.dr_did_att(g.reset_index(drop=True), XC, learner="auto", n_rep=N_REP))

    # ---------------- event study ----------------
    log("== EVENT STUDY ==")
    es = dc.event_study(st, X_RAW, ks=range(-5, 5), base_k=-1, learner="auto", n_rep=N_REP)
    es.to_csv(RES / "event_study.csv", index=False)
    log("event study:\n" + es.round(4).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(-0.5, color="0.6", lw=0.8, ls="--")
    ax.errorbar(es.k, es.theta, yerr=1.645 * es.se, fmt="o", color="#1f4e79", capsize=3)
    ax.set_xlabel("Years relative to enactment of combined reporting")
    ax.set_ylabel("ATT on log credit spread")
    ax.set_title("DML DR-DiD event study (base year k = -1, 90% CI)")
    fig.tight_layout()
    fig.savefig(RES / "event_study.png", dpi=180)

    # ---------------- robustness ----------------
    log("== ROBUSTNESS ==")
    did_block("robust", "effective-date timing", stack(events=ev_eff), ols=False)
    did_block("robust", "drop TX, MI, NY", stack(events=ev_enact[~ev_enact.state.isin(["TX", "MI", "NY"])],
                                                 exclude_states=["TX", "MI", "NY"]), ols=False)
    did_block("robust", "drop 2007-09 cohorts (crisis)", st[~st.cohort.between(2007, 2009)], ols=False)
    did_block("robust", "outcome: spread in bps", st, y="spread_bps", ols=False)
    did_block("robust", "pre-period spread in X", st, pre_y_in_x=True, ols=False)
    did_block("robust", "sample from 2005 (full TRACE)", stack(p=panel[panel.year >= 2005]), ols=False)
    did_block("robust", "threshold: treat>=5%, control<1%",
              stack(treat_min_share=0.05, control_max_share=0.01), ols=False)
    did_block("robust", "threshold: treat>=10%, control<2%",
              stack(treat_min_share=0.10, control_max_share=0.02), ols=False)
    # paper's literal year-by-year rules (presence each year; entrants/leavers dropped)
    cr = dc.new_cr_exposure(expo_f, policy, ev_enact)
    st_y = dc.build_stack(panel, expo_f, ev_enact, cr, pre=5, post=4)
    did_block("robust", "year-by-year exposure (paper rules)", st_y, ols=False)

    # ---------------- heterogeneity ----------------
    log("== HETEROGENEITY ==")
    ld["hi_spread"] = ld.pre_spread_hi
    ld["loss_firm"] = (ld.pre_loss > 0).astype(int)
    ld["big"] = (ld.pre_size > ld.pre_size.median()).astype(int)
    ld["rd_intensive"] = (ld.pre_rd_at > ld.loc[ld.pre_rd_at > 0, "pre_rd_at"].median()).astype(int)
    ld["hi_roa_vol"] = (ld.pre_roa_vol > ld.pre_roa_vol.median()).astype(int)
    ld["hi_lev"] = (ld.pre_book_lev > ld.pre_book_lev.median()).astype(int)
    ld["multi_state"] = (ld.pre_n_states > ld.pre_n_states.median()).astype(int)
    het = []
    for z in ["hi_spread", "loss_firm", "big", "rd_intensive", "hi_roa_vol", "hi_lev", "multi_state"]:
        r = dc.subgroup_diff(ld, XC, z, learner="auto", n_rep=N_REP)
        r["moderator"] = z
        het.append(r)
        log(f"het {z:<13s} z=0 {r[f'theta_{z}=0']: .4f} ({r[f'se_{z}=0']:.4f})  z=1 {r[f'theta_{z}=1']: .4f} "
            f"({r[f'se_{z}=1']:.4f})  diff {r['diff']: .4f} t={r['t_diff']: .2f}")
    het = pd.DataFrame([{"moderator": h["moderator"], "theta_0": h[f"theta_{h['moderator']}=0"],
                         "se_0": h[f"se_{h['moderator']}=0"], "theta_1": h[f"theta_{h['moderator']}=1"],
                         "se_1": h[f"se_{h['moderator']}=1"], "diff": h["diff"], "se_diff": h["se_diff"],
                         "t_diff": h["t_diff"]} for h in het])
    het.to_csv(RES / "heterogeneity.csv", index=False)

    # ---------------- mechanisms (same design, other outcomes) ----------------
    log("== MECHANISM OUTCOMES ==")
    for yv in ["roa_vol", "book_lev", "mkt_lev", "loss", "rd_at"]:
        st_m = st.copy()
        ld_m = dc.long_differences(st_m, X_RAW, y=yv)
        record("mechanism", f"outcome: {yv}", "DML DR-DiD (auto)", dc.dr_did_att(ld_m, XC, learner="auto", n_rep=N_REP))
    log("done.")


if __name__ == "__main__":
    main()
