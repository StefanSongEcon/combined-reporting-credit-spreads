"""
dml_cr.py -- Stacked-cohort double/debiased ML for the effect of state combined
reporting on firm credit spreads (DML version of Welsch 2025, eqs. 5-7).

Pipeline
--------
1. nexus_weight(exposure, policy)        -> firm-year nexus-weighted state variables
2. new_cr_exposure(exposure, policy, ev) -> firm-year NewCRShare, CombinedApptTaxRate
3. build_stack(...)                      -> stacked firm-year panel (treated + clean controls)
4. long_differences(stack, ...)          -> one row per firm x cohort: dY, dT, D, pre-period X
5. Estimators on the long-difference file:
     dr_did_att      binary treatment, orthogonal ATT score     (eq. 5)
     event_study     DR-DiD ATT(k) for each event time k        (eq. 6)
     plr             partially linear model, continuous dose    (eq. 7)
     subgroup_diff   split-sample ATTs and their difference     (cross-sectional tests)
     ols_long_diff   OLS benchmark with industry x cohort dummies

Cross-fitting folds are grouped by firm and SEs are clustered by firm, because a
control firm appears in several stacked cohorts.

Only numpy / pandas / scikit-learn are required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PRE1990_CR_STATES = ["AK", "AZ", "CA", "CO", "HI", "ID", "IL", "KS",
                     "ME", "MN", "MT", "NE", "NH", "ND", "OR", "UT"]


# ----------------------------------------------------------------------------
# 1-2. Exposure construction
# ----------------------------------------------------------------------------
def load_events(path: str, timing: str = "enacted", drop_states=()) -> pd.DataFrame:
    """Read cr_events.csv; `timing` in {'enacted','effective'} sets the cohort year."""
    ev = pd.read_csv(path)
    ev = ev[~ev.state.isin(drop_states)].copy()
    ev["cohort"] = ev[timing].astype(int)
    return ev[["state", "cohort", "enacted", "effective"]]


def nexus_weight(exposure: pd.DataFrame, policy: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Sum_s share_fst * x_st for each policy column (paper's nexus weighting)."""
    m = exposure.merge(policy[["state", "year"] + cols], on=["state", "year"], how="left")
    for c in cols:
        m[c] = m[c] * m["share"]
    out = m.groupby(["firm", "year"], as_index=False)[cols].sum(min_count=1)
    return out.rename(columns={c: f"nw_{c}" for c in cols})


def new_cr_exposure(exposure: pd.DataFrame, policy: pd.DataFrame, events: pd.DataFrame,
                    rate_col: str = "tax_rate") -> pd.DataFrame:
    """Firm-year exposure to *newly adopting* states whose law is in force.

    Returns firm, year, new_cr_share (CombinedAppt), cr_appt_tax_rate (CombinedApptTaxRate),
    n_states, pre1990_cr_share.
    """
    m = exposure.merge(events[["state", "cohort"]], on="state", how="left")
    m = m.merge(policy[["state", "year", rate_col]], on=["state", "year"], how="left")
    in_force = m["cohort"].notna() & (m["year"] >= m["cohort"])
    m["_new"] = np.where(in_force, m["share"], 0.0)
    m["_newrate"] = np.where(in_force, m["share"] * m[rate_col].fillna(0.0), 0.0)
    m["_pre90"] = np.where(m.state.isin(PRE1990_CR_STATES), m["share"], 0.0)
    m["_present"] = (m["share"] > 0).astype(int)
    out = m.groupby(["firm", "year"], as_index=False).agg(
        new_cr_share=("_new", "sum"), cr_appt_tax_rate=("_newrate", "sum"),
        pre1990_cr_share=("_pre90", "sum"), n_states=("_present", "sum"))
    return out


# ----------------------------------------------------------------------------
# 3. Stacked cohorts (Cengiz et al. 2019 / paper Sec. 4.2)
# ----------------------------------------------------------------------------
def build_stack(panel: pd.DataFrame, exposure: pd.DataFrame, events: pd.DataFrame,
                cr: pd.DataFrame, pre: int = 5, post: int = 4,
                treat_min_share: float = 0.0, control_max_share: float = 0.0,
                min_industry_firms: int = 10, verbose: bool = True) -> pd.DataFrame:
    """Return the stacked firm-year panel with columns cohort, rel_time, D, post, T.

    Treated in cohort e : share in S_e at base year (last pre-year with exposure data)
                          > treat_min_share; new_cr_share == 0 in all pre years
                          (not treated earlier); present in S_e in every observed post year.
    Clean control       : new_cr_share <= control_max_share in every window year.
    Firms that are neither (entrants, leavers, earlier-treated) are dropped.
    """
    # only the treatment-defining columns; merge n_states / pre1990_cr_share into `panel`
    # yourself if you want them as covariates
    panel = panel.drop(columns=["new_cr_share", "cr_appt_tax_rate"], errors="ignore").merge(
        cr[["firm", "year", "new_cr_share", "cr_appt_tax_rate"]], on=["firm", "year"], how="left")
    panel[["new_cr_share", "cr_appt_tax_rate"]] = panel[["new_cr_share", "cr_appt_tax_rate"]].fillna(0.0)
    stacks = []
    for e, ev_e in events.groupby("cohort"):
        S_e = set(ev_e.state)
        lo, hi = e - pre, e + post
        win = panel[(panel.year >= lo) & (panel.year <= hi)].copy()
        win["rel_time"] = win.year - e
        win["post"] = (win.year >= e).astype(int)

        # exposure to this cohort's states, by firm-year
        sh = (exposure[exposure.state.isin(S_e) & exposure.year.between(lo, hi)]
              .groupby(["firm", "year"])["share"].sum().rename("share_Se").reset_index())
        win = win.merge(sh, on=["firm", "year"], how="left")
        win["share_Se"] = win["share_Se"].fillna(0.0)

        # base-year presence: last pre-year in which the firm has exposure data
        exp_years = exposure[exposure.year.between(lo, e - 1)][["firm", "year"]].drop_duplicates()
        base = exp_years.groupby("firm").year.max().rename("base_year").reset_index()
        win = win.merge(base, on="firm", how="left")
        base_share = win.loc[win.year == win.base_year, ["firm", "share_Se"]] \
                        .rename(columns={"share_Se": "base_share"}).drop_duplicates("firm")
        win = win.merge(base_share, on="firm", how="left")

        g = win.groupby("firm")
        pre_new = g.apply(lambda d: d.loc[d.post == 0, "new_cr_share"].max(), include_groups=False)
        max_new = g["new_cr_share"].max()
        post_min_Se = g.apply(lambda d: d.loc[d.post == 1, "share_Se"].min(), include_groups=False)
        has_pre = g.apply(lambda d: d.loc[d.post == 0, "y"].notna().any(), include_groups=False)
        has_post = g.apply(lambda d: d.loc[d.post == 1, "y"].notna().any(), include_groups=False)
        bshare = g["base_share"].first().fillna(0.0)
        # firms with no exposure data in the pre-window cannot be classified -> dropped
        has_exp = g["base_year"].first().notna()

        treated = (bshare > treat_min_share) & (pre_new.fillna(0) == 0) & (post_min_Se.fillna(0) > 0)
        control = (max_new <= control_max_share) & (bshare <= control_max_share)
        keep = (treated | control) & has_pre & has_post & has_exp
        status = pd.DataFrame({"D": treated.astype(int), "keep": keep})
        win = win.merge(status, left_on="firm", right_index=True)
        win = win[win.keep].drop(columns=["keep", "base_year"])
        win["cohort"] = e
        # continuous dose: only treated carry CombinedApptTaxRate (controls are 0 by construction
        # when control_max_share == 0; with a positive threshold set their dose to 0 too)
        win["T"] = np.where(win.D == 1, win["cr_appt_tax_rate"], 0.0)
        win["A"] = np.where(win.D == 1, win["new_cr_share"], 0.0)
        if verbose:
            nt = win.loc[win.D == 1, "firm"].nunique()
            nc = win.loc[win.D == 0, "firm"].nunique()
            print(f"cohort {e} {sorted(S_e)}: treated firms {nt:5d}, clean controls {nc:5d}")
        stacks.append(win)
    st = pd.concat(stacks, ignore_index=True)
    # drop thin industries (counted on unique firms in the stacked sample)
    ind_n = st.groupby("industry").firm.nunique()
    st = st[st.industry.isin(ind_n[ind_n >= min_industry_firms].index)]
    st["unit"] = st.firm.astype(str) + "_" + st.cohort.astype(str)
    return st


# ----------------------------------------------------------------------------
# 4. Long differences within firm x cohort
# ----------------------------------------------------------------------------
def long_differences(st: pd.DataFrame, x_cols: list[str], y: str = "y",
                     k: int | None = None, base_k: int = -1,
                     include_pre_y: bool = False) -> pd.DataFrame:
    """One row per firm x cohort.

    k=None : dY = mean(y | post) - mean(y | pre)    (eq. 5 analogue)
    k=int  : dY = y(e+k) - y(e+base_k)              (eq. 6 analogue)
    X      : pre-period means of x_cols (pre-determined covariates), plus
             industry and cohort indicators.
    """
    pre = st[st.post == 0]
    X = pre.groupby("unit")[x_cols].mean().add_prefix("pre_")
    if include_pre_y:
        X["pre_y"] = pre.groupby("unit")[y].mean()
    meta = st.groupby("unit").agg(firm=("firm", "first"), cohort=("cohort", "first"),
                                  industry=("industry", "first"), D=("D", "first"))
    if k is None:
        ypost = st[st.post == 1].groupby("unit")[y].mean()
        ypre = pre.groupby("unit")[y].mean()
        dY = (ypost - ypre).rename("dY")
    else:
        a = st[st.rel_time == k].set_index("unit")[y]
        b = st[st.rel_time == base_k].set_index("unit")[y]
        dY = (a - b).rename("dY")
    dT = (st[st.post == 1].groupby("unit")["T"].mean()
          - pre.groupby("unit")["T"].mean()).rename("dT")
    dA = (st[st.post == 1].groupby("unit")["A"].mean()
          - pre.groupby("unit")["A"].mean()).rename("dA")
    ld = meta.join([dY, dT, dA, X], how="left").dropna(subset=["dY"])
    ld = ld.reset_index()
    return ld


def design_matrix(ld: pd.DataFrame, x_cols: list[str]) -> np.ndarray:
    """Numeric pre-period covariates (median-imputed + missing flags) + industry & cohort dummies."""
    Xn = ld[x_cols].copy()
    for c in x_cols:
        if Xn[c].isna().any():
            Xn[c + "_miss"] = Xn[c].isna().astype(float)
            Xn[c] = Xn[c].fillna(Xn[c].median())
    dums = pd.get_dummies(ld[["industry", "cohort"]].astype(str), drop_first=True, dtype=float)
    return pd.concat([Xn, dums], axis=1).to_numpy(dtype=float)


# ----------------------------------------------------------------------------
# 5. Learners and cross-fitting utilities
# ----------------------------------------------------------------------------
def make_learners(kind: str = "gbm", seed: int = 0):
    """Return (regressor, classifier) nuisance learners."""
    from sklearn.linear_model import RidgeCV
    if kind == "gbm":
        # capacity adapts to sample size through early stopping on a 15% validation split
        kw = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                  n_iter_no_change=25, random_state=seed)
        reg, clf = HistGradientBoostingRegressor(**kw), HistGradientBoostingClassifier(**kw)
    elif kind == "rf":
        reg = RandomForestRegressor(n_estimators=500, min_samples_leaf=10, max_features=0.5,
                                    n_jobs=4, random_state=seed)
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=10, max_features=0.5,
                                     n_jobs=4, random_state=seed)
    elif kind == "lasso":
        reg = make_pipeline(StandardScaler(), LassoCV(cv=5, n_alphas=50, random_state=seed))
        clf = make_pipeline(StandardScaler(),
                            LogisticRegressionCV(Cs=20, cv=5, penalty="l1", solver="saga",
                                                 max_iter=5000, random_state=seed))
    elif kind == "logit":   # ridge / L2-logit: smooth, well-calibrated propensities
        reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
        clf = make_pipeline(StandardScaler(),
                            LogisticRegressionCV(Cs=20, cv=5, penalty="l2", max_iter=5000))
    else:
        raise ValueError(kind)
    return reg, clf


AUTO_CANDIDATES = ("logit", "gbm", "rf")
PLR_CANDIDATES = ("logit", "gbm")


def group_folds(groups: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Random fold id per row, constant within group (firm)."""
    ug = np.unique(groups)
    perm = rng.permutation(len(ug))
    fold_of = dict(zip(ug[perm], np.arange(len(ug)) % k))
    return np.array([fold_of[g] for g in groups])


def cluster_se(phi: np.ndarray, groups: np.ndarray) -> float:
    s = pd.Series(phi).groupby(groups).sum().to_numpy()
    return float(np.sqrt(np.sum(s ** 2)) / len(phi))


def _median_aggregate(thetas, ses):
    thetas, ses = np.asarray(thetas), np.asarray(ses)
    th = float(np.median(thetas))
    se = float(np.sqrt(np.median(ses ** 2 + (thetas - th) ** 2)))
    return th, se


# ----------------------------------------------------------------------------
# 5a. DR-DiD ATT (binary treatment)
# ----------------------------------------------------------------------------
def _oof(X, Y, D, folds, kind, seed, n_folds):
    reg, clf = make_learners(kind, seed)
    m_hat, l_hat = np.zeros(len(Y)), np.zeros(len(Y))
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        m_hat[te] = clone(clf).fit(X[tr], D[tr]).predict_proba(X[te])[:, 1]
        ctrl = tr & (D == 0)
        l_hat[te] = clone(reg).fit(X[ctrl], Y[ctrl]).predict(X[te])
    return m_hat, l_hat


def dr_did_att(ld: pd.DataFrame, x_cols: list[str], learner: str = "gbm", n_folds: int = 5,
               n_rep: int = 10, trim: float = 0.01, seed: int = 0, normalized: bool = True,
               y: str = "dY", d: str = "D", return_phi: bool = False) -> dict:
    """Doubly-robust DiD ATT with cross-fitting (Chang 2020; Sant'Anna & Zhao 2020).

    normalized=True uses Hajek-normalized control weights (Sant'Anna-Zhao "improved"
    DR form), which is far more stable when propensities approach 1.
    learner="auto": in each repetition, fit AUTO_CANDIDATES on the same folds and use the
    propensity learner with the lowest out-of-fold log-loss and the outcome learner with
    the lowest out-of-fold MSE among controls.
    """
    X = design_matrix(ld, x_cols)
    Y = ld[y].to_numpy(float)
    D = ld[d].to_numpy(int)
    G = ld["firm"].to_numpy()
    rng = np.random.default_rng(seed)
    thetas, ses, phis, mhats, chosen = [], [], [], [], []
    diag = {"r2_l0": [], "logloss_m": [], "auc_m": []}
    c0 = D == 0
    for r in range(n_rep):
        folds = group_folds(G, n_folds, rng)
        kinds = AUTO_CANDIDATES if learner == "auto" else (learner,)
        fits = {k: _oof(X, Y, D, folds, k, seed + r, n_folds) for k in kinds}
        km = min(kinds, key=lambda k: log_loss(D, np.clip(fits[k][0], 1e-6, 1 - 1e-6)))
        kl = min(kinds, key=lambda k: np.mean((Y[c0] - fits[k][1][c0]) ** 2))
        m_hat, l_hat = fits[km][0], fits[kl][1]
        chosen.append(f"m:{km}/l:{kl}")
        diag["logloss_m"].append(log_loss(D, np.clip(m_hat, 1e-6, 1 - 1e-6)))
        diag["auc_m"].append(roc_auc_score(D, m_hat))
        diag["r2_l0"].append(1 - np.mean((Y[c0] - l_hat[c0]) ** 2) / np.var(Y[c0]))
        m_t = np.clip(m_hat, trim, 1 - trim)
        w = m_t / (1 - m_t)
        res = Y - l_hat
        p = D.mean()
        if normalized:
            wc = (1 - D) * w
            th1 = np.sum(D * res) / D.sum()
            th0 = np.sum(wc * res) / wc.sum()
            theta = th1 - th0
            phi = D * (res - th1) / p - wc * (res - th0) / wc.mean()
        else:
            num = D * res - (1 - D) * w * res
            theta = num.sum() / D.sum()
            phi = (num - D * theta) / p
        thetas.append(theta)
        ses.append(cluster_se(phi, G))
        phis.append(phi)
        mhats.append(m_hat)
    th, se = _median_aggregate(thetas, ses)
    m_bar = np.mean(mhats, axis=0)
    out = {"theta": th, "se": se, "t": th / se, "n_units": len(Y), "n_treated": int(D.sum()),
           "n_firms": int(pd.Series(G).nunique()),
           "share_m_at_bounds": float(np.mean((m_bar < trim) | (m_bar > 1 - trim))),
           "r2_l0": float(np.median(diag["r2_l0"])), "auc_m": float(np.median(diag["auc_m"])),
           "logloss_m": float(np.median(diag["logloss_m"])), "m_hat": m_bar,
           "learners": pd.Series(chosen).mode().iloc[0]}
    if return_phi:
        j = int(np.argmin(np.abs(np.asarray(thetas) - th)))
        out["phi"] = phis[j]
    return out


def balance_table(ld: pd.DataFrame, x_cols: list[str], m_hat: np.ndarray, trim: float = 0.01):
    """Standardized mean differences, raw and with ATT weights m/(1-m) on controls."""
    D = ld["D"].to_numpy(int)
    m = np.clip(m_hat, trim, 1 - trim)
    w = np.where(D == 1, 1.0, m / (1 - m))
    rows = []
    for c in x_cols:
        x = ld[c].to_numpy(float)
        ok = ~np.isnan(x)
        t, cc = ok & (D == 1), ok & (D == 0)
        sd = np.sqrt((np.var(x[t]) + np.var(x[cc])) / 2) or 1.0
        raw = (x[t].mean() - x[cc].mean()) / sd
        wtd = (x[t].mean() - np.average(x[cc], weights=w[cc])) / sd
        rows.append((c, raw, wtd))
    return pd.DataFrame(rows, columns=["covariate", "smd_raw", "smd_weighted"])


# ----------------------------------------------------------------------------
# 5b. Event study
# ----------------------------------------------------------------------------
def event_study(st: pd.DataFrame, x_cols_raw: list[str], ks=range(-5, 5), base_k: int = -1,
                **kw) -> pd.DataFrame:
    rows = []
    for k in ks:
        if k == base_k:
            rows.append((k, 0.0, 0.0, np.nan))
            continue
        ld = long_differences(st, x_cols_raw, k=k, base_k=base_k)
        xc = [f"pre_{c}" for c in x_cols_raw]
        r = dr_did_att(ld, xc, **kw)
        rows.append((k, r["theta"], r["se"], r["n_units"]))
    return pd.DataFrame(rows, columns=["k", "theta", "se", "n_units"])


# ----------------------------------------------------------------------------
# 5c. Partially linear model (continuous dose)
# ----------------------------------------------------------------------------
def plr(ld: pd.DataFrame, x_cols: list[str], treat: str = "dT", y: str = "dY",
        learner: str = "gbm", n_folds: int = 5, n_rep: int = 10, seed: int = 0) -> dict:
    """Partially linear model; learner="auto" picks, per nuisance, the candidate with the
    lowest out-of-fold MSE."""
    X = design_matrix(ld, x_cols)
    Y = ld[y].to_numpy(float)
    T = ld[treat].to_numpy(float)
    G = ld["firm"].to_numpy()
    rng = np.random.default_rng(seed)
    thetas, ses, r2l, r2m, chosen = [], [], [], [], []
    for r in range(n_rep):
        folds = group_folds(G, n_folds, rng)
        # random forests are excluded from the PLR candidates: in simulations with smooth
        # trend confounding they had the lowest MSE but left a 15% bias in theta
        kinds = PLR_CANDIDATES if learner == "auto" else (learner,)
        preds = {}
        for k in kinds:
            reg, _ = make_learners(k, seed + r)
            lh, mh = np.zeros(len(Y)), np.zeros(len(Y))
            for f in range(n_folds):
                tr, te = folds != f, folds == f
                lh[te] = clone(reg).fit(X[tr], Y[tr]).predict(X[te])
                mh[te] = clone(reg).fit(X[tr], T[tr]).predict(X[te])
            preds[k] = (lh, mh)
        # one learner for both nuisances: lowest sum of relative out-of-fold MSEs
        # (mixing learners across nuisances produced correlated errors in simulations)
        kl = km = min(kinds, key=lambda k: np.mean((Y - preds[k][0]) ** 2) / np.var(Y)
                      + np.mean((T - preds[k][1]) ** 2) / np.var(T))
        chosen.append(f"m:{km}/l:{kl}")
        u, v = Y - preds[kl][0], T - preds[km][1]
        theta = np.sum(u * v) / np.sum(v * v)
        phi = (u - theta * v) * v / np.mean(v * v)
        thetas.append(theta)
        ses.append(cluster_se(phi, G))
        r2l.append(1 - np.mean(u ** 2) / np.var(Y))
        r2m.append(1 - np.mean(v ** 2) / np.var(T))
    th, se = _median_aggregate(thetas, ses)
    return {"theta": th, "se": se, "t": th / se, "n_units": len(Y),
            "r2_l": float(np.median(r2l)), "r2_m": float(np.median(r2m)),
            "learners": pd.Series(chosen).mode().iloc[0]}


# ----------------------------------------------------------------------------
# 5d. Heterogeneity: split treated by a pre-period moderator, keep all controls
# ----------------------------------------------------------------------------
def subgroup_diff(ld: pd.DataFrame, x_cols: list[str], z: str, **kw) -> dict:
    """ATT for treated with z==1 and z==0 (same controls); firm-clustered test of the gap."""
    out, phis = {}, {}
    for v in (0, 1):
        keep = (ld.D == 0) | (ld[z] == v)
        sub = ld[keep].reset_index(drop=True)
        r = dr_did_att(sub, x_cols, return_phi=True, **kw)
        out[f"theta_{z}={v}"], out[f"se_{z}={v}"] = r["theta"], r["se"]
        # map phi back to the full index (0 for units outside the subsample)
        full = np.zeros(len(ld))
        # rescale so that cluster_se(full, G) on N=len(ld) equals the subsample SE
        full[np.flatnonzero(keep.to_numpy())] = r["phi"] * len(ld) / len(sub)
        phis[v] = full
    diff = out[f"theta_{z}=1"] - out[f"theta_{z}=0"]
    se = cluster_se(phis[1] - phis[0], ld["firm"].to_numpy())
    out.update({"diff": diff, "se_diff": se, "t_diff": diff / se})
    return out


# ----------------------------------------------------------------------------
# 5e. OLS benchmark: long-difference regression with industry x cohort dummies
# ----------------------------------------------------------------------------
def ols_long_diff(ld: pd.DataFrame, x_cols: list[str], treat: str = "D", y: str = "dY") -> dict:
    Xn = ld[x_cols].fillna(ld[x_cols].median())
    cell = pd.get_dummies(ld.industry.astype(str) + "_" + ld.cohort.astype(str),
                          drop_first=True, dtype=float)
    Z = np.column_stack([np.ones(len(ld)), ld[treat].to_numpy(float), Xn.to_numpy(float),
                         cell.to_numpy()])
    Y = ld[y].to_numpy(float)
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    e = Y - Z @ beta
    ZtZi = np.linalg.pinv(Z.T @ Z)
    G = ld["firm"].to_numpy()
    sc = pd.DataFrame(Z * e[:, None]).groupby(G).sum().to_numpy()
    V = ZtZi @ (sc.T @ sc) @ ZtZi
    return {"theta": float(beta[1]), "se": float(np.sqrt(V[1, 1])),
            "t": float(beta[1] / np.sqrt(V[1, 1])), "n_units": len(Y)}


# ----------------------------------------------------------------------------
# 3b. Stacked cohorts with exposure frozen at its pre-period average
#     (intent-to-treat; robust to year-to-year noise in text-based exposure)
# ----------------------------------------------------------------------------
def build_stack_frozen(panel: pd.DataFrame, exposure: pd.DataFrame, policy: pd.DataFrame,
                       events: pd.DataFrame, pre: int = 5, post: int = 4,
                       treat_min_share: float = 0.0, control_max_share: float = 0.0,
                       exclude_states=(), rate_col: str = "tax_rate",
                       min_industry_firms: int = 10, verbose: bool = True) -> pd.DataFrame:
    """Stacked panel where each firm's state shares are fixed at their pre-period mean.

    For cohort e with states S_e and window [e-pre, e+post], with Ebar_fs the mean share
    over the firm's 10-K years in [e-pre, e-1] (unmentioned state = 0 that year):
      earlier  = sum of Ebar over states adopting before e      (already treated)
      later    = sum of Ebar over states adopting in (e, e+post] (treated inside window)
      treated  : share_Se > treat_min_share and earlier <= control_max_share
      control  : share_Se, earlier, later all <= control_max_share
      excluded : any firm with Ebar > control_max_share in `exclude_states`
    Dose T_ft = sum over newly adopting states in force at t of Ebar_fs * tax_rate_st
    (treated only); A_ft = the same without the tax rate.
    """
    coh = dict(zip(events.state, events.cohort))
    rates = policy.set_index(["state", "year"])[rate_col]
    stacks = []
    for e, ev_e in events.groupby("cohort"):
        S_e = set(ev_e.state)
        lo, hi = e - pre, e + post
        ex = exposure[exposure.year.between(lo, e - 1)]
        if ex.empty:
            continue
        nyrs = ex.groupby("firm").year.nunique()
        eb = ex.groupby(["firm", "state"]).share.sum().div(nyrs, level="firm").rename("ebar").reset_index()
        eb["cohort_s"] = eb.state.map(coh)
        f = pd.DataFrame(index=nyrs.index)
        f["share_Se"] = eb[eb.state.isin(S_e)].groupby("firm").ebar.sum()
        f["earlier"] = eb[eb.cohort_s < e].groupby("firm").ebar.sum()
        f["later"] = eb[(eb.cohort_s > e) & (eb.cohort_s <= hi)].groupby("firm").ebar.sum()
        f["excl"] = eb[eb.state.isin(exclude_states)].groupby("firm").ebar.sum()
        f = f.fillna(0.0)
        ok = f.excl <= control_max_share
        treated = ok & (f.share_Se > treat_min_share) & (f.earlier <= control_max_share)
        control = ok & (f.share_Se <= control_max_share) & (f.earlier <= control_max_share) \
            & (f.later <= control_max_share)
        status = pd.Series(np.where(treated, 1, np.where(control, 0, -1)), index=f.index, name="D")
        status = status[status >= 0]

        win = panel[panel.year.between(lo, hi) & panel.firm.isin(status.index)].copy()
        win = win.merge(status, left_on="firm", right_index=True)
        win["rel_time"] = win.year - e
        win["post"] = (win.year >= e).astype(int)
        has = win.dropna(subset=["y"]).groupby("firm").post.agg(["min", "max"])
        good = has[(has["min"] == 0) & (has["max"] == 1)].index
        win = win[win.firm.isin(good)]

        # frozen-share dose for treated firms
        newst = eb[eb.cohort_s.notna() & eb.firm.isin(status[status == 1].index)]
        dose = []
        for t in range(lo, hi + 1):
            act = newst[newst.cohort_s <= t]
            if act.empty:
                continue
            r = act.state.map(lambda s: rates.get((s, t), np.nan)).fillna(0.0)
            d = pd.DataFrame({"firm": act.firm, "T": act.ebar * r, "A": act.ebar})
            d = d.groupby("firm")[["T", "A"]].sum().reset_index()
            d["year"] = t
            dose.append(d)
        if dose:
            win = win.merge(pd.concat(dose), on=["firm", "year"], how="left")
        else:
            win["T"], win["A"] = 0.0, 0.0
        win[["T", "A"]] = win[["T", "A"]].fillna(0.0)
        win.loc[win.D == 0, ["T", "A"]] = 0.0
        win["share_Se"] = win.firm.map(f.share_Se)
        win["cohort"] = e
        if verbose:
            nt = win.loc[win.D == 1, "firm"].nunique()
            nc = win.loc[win.D == 0, "firm"].nunique()
            print(f"cohort {e} {sorted(S_e)}: treated firms {nt:5d}, clean controls {nc:5d}")
        stacks.append(win)
    st = pd.concat(stacks, ignore_index=True)
    ind_n = st.groupby("industry").firm.nunique()
    st = st[st.industry.isin(ind_n[ind_n >= min_industry_firms].index)]
    st["unit"] = st.firm.astype(str) + "_" + st.cohort.astype(str)
    return st


# ----------------------------------------------------------------------------
# 5f. Paper's stacked OLS (eq. 5 / 7): firm x cohort FE + industry x year x cohort FE
# ----------------------------------------------------------------------------
def _demean_2way(df: pd.DataFrame, cols: list[str], g1: str, g2: str, tol=1e-8, maxit=500):
    X = df[cols].to_numpy(float).copy()
    a, b = df[g1].to_numpy(), df[g2].to_numpy()
    for _ in range(maxit):
        old = X.copy()
        X -= pd.DataFrame(X).groupby(a).transform("mean").to_numpy()
        X -= pd.DataFrame(X).groupby(b).transform("mean").to_numpy()
        if np.max(np.abs(X - old)) < tol:
            break
    return X


def stacked_fe_ols(st: pd.DataFrame, treat: str = "Dpost", controls: list[str] = (),
                   y: str = "y") -> dict:
    """OLS of y on treatment with firm-cohort and industry-year-cohort FE; SE clustered by firm."""
    d = st.dropna(subset=[y] + list(controls)).copy()
    if treat == "Dpost":
        d["Dpost"] = d.D * d.post
    d["fc"] = d.unit
    d["iyc"] = d.industry.astype(str) + "_" + d.year.astype(str) + "_" + d.cohort.astype(str)
    cols = [y, treat] + list(controls)
    Z = _demean_2way(d, cols, "fc", "iyc")
    Y, X = Z[:, 0], Z[:, 1:]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    e = Y - X @ beta
    XtXi = np.linalg.pinv(X.T @ X)
    sc = pd.DataFrame(X * e[:, None]).groupby(d.firm.to_numpy()).sum().to_numpy()
    V = XtXi @ (sc.T @ sc) @ XtXi
    return {"theta": float(beta[0]), "se": float(np.sqrt(V[0, 0])),
            "t": float(beta[0] / np.sqrt(V[0, 0])), "n_units": int(len(d))}
