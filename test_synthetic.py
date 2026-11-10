"""
Simulation check for dml_cr.py.

World: firms spread over states; large firms operate in more states (so they are more
likely to be treated) AND have a different, nonlinear spread trend (confounding).
True effect: log spread rises by THETA per percentage point of CombinedApptTaxRate.
The script checks that DML recovers THETA (PLR) and the implied ATT (DR-DiD), and
shows the bias of naive and linear-OLS long differences.

Run:  python3 test_synthetic.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import dml_cr as dc

HERE = Path(__file__).resolve().parent
THETA = 0.03          # effect on log spread per pp of CombinedApptTaxRate
N_FIRMS = 3000
YEARS = np.arange(2000, 2025)
STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL",
          "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE",
          "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
          "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]


def simulate(seed=1):
    rng = np.random.default_rng(seed)
    # state policy: top rate and one other policy variable
    rate = dict(zip(STATES, rng.uniform(4, 10, len(STATES))))
    policy = pd.DataFrame([(s, t, rate[s] + rng.normal(0, 0.1), int(rng.random() < 0.5))
                           for s in STATES for t in YEARS],
                          columns=["state", "year", "tax_rate", "throwback"])
    # state popularity (big states attract more firms)
    pop = rng.dirichlet(np.ones(len(STATES)) * 0.6)

    size0 = rng.normal(0, 1, N_FIRMS)
    lev0 = np.clip(0.25 + 0.1 * rng.normal(size=N_FIRMS) - 0.04 * size0, 0.01, 0.9)
    industry = rng.integers(0, 12, N_FIRMS)
    n_states = 1 + rng.poisson(np.exp(0.2 + 0.7 * size0))
    n_states = np.minimum(n_states, 30)

    exp_rows, firm_states = [], {}
    for f in range(N_FIRMS):
        st = rng.choice(STATES, size=n_states[f], replace=False, p=pop)
        sh = rng.dirichlet(np.ones(len(st)) * 0.8)
        firm_states[f] = (st, sh)
        for t in YEARS:
            for s, a in zip(st, sh):
                exp_rows.append((f, t, s, a))
    exposure = pd.DataFrame(exp_rows, columns=["firm", "year", "state", "share"])

    events = dc.load_events(HERE / "cr_events.csv", timing="enacted", drop_states=["NM"])
    cr = dc.new_cr_exposure(exposure, policy, events)

    # firm-year panel
    rows = []
    trend = 0.02 * np.tanh(2 * size0) + 0.03 * (lev0 > 0.3) * (size0 < 0)  # nonlinear confounder
    alpha = rng.normal(5.5, 0.5, N_FIRMS)
    yshock = dict(zip(YEARS, rng.normal(0, 0.1, len(YEARS))))
    ishock = {(i, t): rng.normal(0, 0.05) for i in range(12) for t in YEARS}
    for f in range(N_FIRMS):
        for t in YEARS:
            size = size0[f] + 0.02 * (t - 2000) + rng.normal(0, 0.05)
            lev = np.clip(lev0[f] + rng.normal(0, 0.02), 0, 1)
            y0 = (alpha[f] + yshock[t] + ishock[(industry[f], t)] + trend[f] * (t - 2000)
                  + rng.normal(0, 0.08))
            rows.append((f, t, industry[f], size, lev, y0))
    panel = pd.DataFrame(rows, columns=["firm", "year", "industry", "size", "lev", "y0"])
    panel = panel.merge(cr[["firm", "year", "cr_appt_tax_rate", "n_states"]],
                        on=["firm", "year"], how="left")
    panel["y"] = panel["y0"] + THETA * panel["cr_appt_tax_rate"]
    nw = dc.nexus_weight(exposure, policy, ["tax_rate", "throwback"])
    panel = panel.merge(nw, on=["firm", "year"], how="left")
    panel = panel.drop(columns=["cr_appt_tax_rate"])
    # unbalanced panel: drop 10% of outcomes
    panel.loc[rng.random(len(panel)) < 0.10, "y"] = np.nan
    return panel, exposure, policy, events, cr


def fmt(name, r, truth=None):
    s = f"{name:<34s} theta={r['theta']: .4f}  se={r['se']:.4f}  t={r['t']: .2f}"
    if truth is not None:
        s += f"   [truth {truth:.4f}]"
    return s


def main():
    t0 = time.time()
    panel, exposure, policy, events, cr = simulate()
    print(f"simulated in {time.time() - t0:.1f}s; panel {panel.shape}")

    t1 = time.time()
    x_raw = ["size", "lev", "nw_tax_rate", "nw_throwback", "n_states"]
    st = dc.build_stack(panel, exposure, events, cr, pre=5, post=4, min_industry_firms=10)
    ld = dc.long_differences(st, x_raw)
    xc = [f"pre_{c}" for c in x_raw]
    print(f"stacked units {len(ld)} (treated {int(ld.D.sum())}), built in {time.time() - t1:.1f}s")

    # truth for the binary ATT: theta * mean change in dose among treated units
    att_true = THETA * ld.loc[ld.D == 1, "dT"].mean()

    t2 = time.time()
    naive = dc.ols_long_diff(ld, [], treat="D")
    ols = dc.ols_long_diff(ld, xc, treat="D")
    att = dc.dr_did_att(ld, xc, learner="gbm", n_rep=3, return_phi=True)
    att_rf = dc.dr_did_att(ld, xc, learner="lasso", n_rep=2)
    print(fmt("Long-diff OLS, no X (naive)", naive, att_true))
    print(fmt("Long-diff OLS, linear X", ols, att_true))
    print(fmt("DML DR-DiD ATT (gbm)", att, att_true))
    print(fmt("DML DR-DiD ATT (lasso)", att_rf, att_true))
    print(f"   nuisance: AUC(m)={att['auc_m']:.3f}  R2(l0)={att['r2_l0']:.3f}  "
          f"share m at trim bounds={att['share_m_at_bounds']:.3f}")

    ols_c = dc.ols_long_diff(ld, xc, treat="dT")
    pl = dc.plr(ld, xc, treat="dT", learner="gbm", n_rep=3)
    print(fmt("Long-diff OLS dose, linear X", ols_c, THETA))
    print(fmt("DML PLR dose (gbm)", pl, THETA))
    print(f"   nuisance: R2(l)={pl['r2_l']:.3f}  R2(m_T)={pl['r2_m']:.3f}")
    print(f"estimators ran in {time.time() - t2:.1f}s")

    bal = dc.balance_table(ld, xc, att["m_hat"])
    print("\nBalance (standardized mean differences):\n", bal.round(3).to_string(index=False))

    ld["big"] = (ld["pre_size"] > ld["pre_size"].median()).astype(int)
    sg = dc.subgroup_diff(ld, xc, "big", learner="gbm", n_rep=2)
    print("\nSubgroup (pre-period size above median):",
          {k: round(v, 4) for k, v in sg.items()})

    t3 = time.time()
    es = dc.event_study(st, x_raw, ks=range(-5, 5), learner="gbm", n_rep=2)
    print("\nEvent study (DML DR-DiD ATT(k), base k=-1):\n", es.round(4).to_string(index=False))
    print(f"event study ran in {time.time() - t3:.1f}s; total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
