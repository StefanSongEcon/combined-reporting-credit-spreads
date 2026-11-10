# Combined Reporting and Corporate Credit Spreads: a DML Version of Welsch (2025)

Base paper: Welsch, A. (2025), "Riskier Together? The Effect of Combined Tax Reporting on
Corporate Risk-Taking," SSRN 5649711.

This note keeps the paper's identification strategy: staggered state adoption of mandatory
combined reporting, a stacked-cohort design with clean controls, nexus-weighted exposure, and
binary and continuous treatments. It makes two changes:

1. **Outcome.** The outcome is the firm-level corporate bond credit spread instead of ROA volatility.
2. **Estimator.** Double/debiased machine learning (DML) replaces "entropy balancing + OLS with controls."
   The ML nuisance functions take over the jobs that entropy balancing (who gets compared with whom)
   and the linear control vector (how covariates shift the outcome trend) do in the paper.

---

## 1. Question and hypotheses

**Question.** When a state switches from separate to combined reporting, do credit spreads change
for firms with nexus in that state, relative to clean control firms?

The sign is ambiguous before looking at the data, so the hypothesis is stated as a null, as in the paper:

| Channel | Mechanism | Predicted effect on spreads |
|---|---|---|
| Risk-taking (paper's main result) | Firms take on higher-variance projects, so asset volatility and default probability rise (Merton) | + |
| Loss-offset insurance | One affiliate's losses offset another affiliate's profits right away. The state shares downside losses, so after-tax cash flow in bad states improves. | − |
| Tax burden | Most treated firms' state tax liability *rises* (paper App. E), which lowers free cash flow | + (partly offset by a larger interest tax shield) |
| Lost income shifting | IP-intensive firms lose low-tax-state royalty structures | + for patent-intensive firms |

Creditors hold a short put on firm assets, so they care mostly about the *downside*. The
risk-taking channel (more variance) and the insurance channel (a truncated after-tax downside)
can therefore push in opposite directions. This is the main contribution relative to the paper:
a market price that combines both effects.

**Hypothesis (null form).** Combined reporting has no effect on firm credit spreads.

---

## 2. What is kept from the paper and what changes

| Element | Welsch (2025) | This project |
|---|---|---|
| Shock | 11 states + DC, 2006–2019 (App. D) | Same list. NM (effective 2020) can be added because spreads run to 2025 |
| Design | Stacked cohorts, window [e−5, e+4], clean controls | Same |
| Treated | Firm has establishments in an adopting state (Data Axle); excludes firms that enter or leave | Same rule, applied to the exposure panel (§4) |
| Earlier-treated firms | Dropped from later cohorts | Same |
| Fixed effects | firm×cohort, industry×year×cohort | Firm×cohort FE → long differences within firm×cohort; industry×cohort trends → industry and cohort features inside the nuisance functions (plus an OLS benchmark with explicit dummies) |
| Covariate adjustment | Entropy balancing on pre-period means/variances + linear controls | DML: propensity m(X) and outcome-trend ℓ(X) learned flexibly, combined in a Neyman-orthogonal score |
| Binary treatment, eq. (5) | OLS β₁ | DML doubly-robust DiD ATT (Chang 2020; Sant'Anna & Zhao 2020) |
| Event study, eq. (6) | OLS βₖ | DML DR-DiD ATT(k) for each k, with base year e−1 |
| Continuous treatment, eq. (7) | OLS on CombinedApptTaxRate | DML partially linear model in long differences |
| Cross-sectional tests | Split samples / interactions | Subgroup DML ATTs with a cluster-robust test of the difference |
| Oster (2019) test | Yes | Chernozhukov–Cinelli–Newey–Sharma–Syrgkanis (2022) omitted-variable-bias bounds for DML |
| SEs | Clustered by firm×cohort | Clustered by **firm** (a stacked control firm shows up in several cohorts), with cross-fitting folds also split by firm |

---

## 3. Events, timing, and sample period

| State | Enacted | Effective | Notes |
|---|---|---|---|
| VT | 2004-06-07 | 2006-01-01 | TRACE pre-period is thin (see below) |
| TX | 2006-05-23 | 2007-01-01 | Margin tax replaced franchise tax at the same time (drop in robustness) |
| MI | 2007-07-12 | 2008-01-01 | SBT → MBT at the same time (drop in robustness) |
| WV | 2007-04-04 | 2009-01-01 | |
| MA | 2008-07-02 | 2009-01-01 | |
| WI | 2009-02-19 | 2009-01-01 | |
| DC | 2011-09-14 | 2011-01-01 | Retroactive |
| NY | 2014-03-31 | 2015-01-01 | Partial combination from 2007 (biases toward 0) |
| RI | 2014-06-19 | 2015-01-01 | |
| CT | 2015-06-30 | 2016-01-01 | |
| KY | 2018-04-27 | 2019-01-01 | Paper drops this cohort (forward ROA-vol window). Spreads have no such constraint, so keep it |
| NJ | 2018-07-01 | 2019-01-01 | Same as KY |
| NM | 2019-04-04 | 2020-01-01 | Extension only. The post-window overlaps COVID |

Pre-1990 combined reporting states (AK AZ CA CO HI ID IL KS ME MN MT NE NH ND OR UT) are never
"newly treated." Firms located only in these states and in separate-reporting states are clean controls.

**Event date: enactment versus effective date.** ROA volatility is realized after the law takes
effect. Bond prices react when the news arrives. **Main spec:** event year = enactment year, base
year = enactment year − 1. **Alternative:** the effective year, which matches the paper. The gap
matters for VT (2 years), WV (2 years), and roughly one year for most others. The event-study
pre-period coefficients for the effective-date timing will show anticipation if the enactment
timing is the right one.

**TRACE coverage.** Dissemination was phased in from July 2002 and reached full coverage in
February 2005. The current firm-year panel (`empirical/temp/gvkey_year_credit_spreads.dta`) has
237 firms in 2002 and about 600 per year from 2005. Consequences:
- The VT cohort (enacted 2004) has no usable pre-period. Drop it, or use only k ≥ −2.
- Robustness: start the panel in 2005.
- An extension uses weekly TRACE spreads (`issuer_week_credit_spreads.dta`) for a high-frequency
  event study around enactment dates. This is the cleanest test for a price outcome.

**Sample filters** (as in the paper): drop financials (NAICS 52), utilities (22), and the public
sector (92); drop industries with fewer than 10 firms; require at least one outcome observation
pre and post in each cohort window; drop firms that enter or leave an adopting state around the event.

---

## 4. Treatment definitions

Let $a_{fst}$ be firm $f$'s activity share in state $s$ in year $t$. As in the paper,
$a_{fst} = 0.5\,\frac{\text{emp}_{fst}}{\text{emp}_{f,US,t}} + 0.5\,\frac{\text{sales}_{fst}}{\text{sales}_{f,US,t}}$.
Let $S_e$ be the states adopting in cohort $e$ and $\mathcal N$ all newly adopting states.

- **NewCRShare**$_{ft} = \sum_{s\in\mathcal N,\ e(s)\le t} a_{fst}$, the exposure to new laws in force.
- **Treated in cohort e:** $\sum_{s\in S_e} a_{fs,e-1} > 0$ (presence at base); NewCRShare = 0 in
  all pre-years (not treated earlier); positive presence in $S_e$ in every post year (no exit).
- **Clean control for cohort e:** NewCRShare$_{ft}=0$ for every $t\in[e-5, e+4]$. This covers
  never-treated firms and not-yet-treated firms. It excludes entrants by construction.
- **CombinedApptTaxRate**$_{ft} = \sum_{s\in\mathcal N,\ e(s)\le t} a_{fst}\,\tau_{st}$ (paper's eq. 7
  treatment, in percentage points).
- **CombinedAppt**$_{ft}$ = NewCRShare$_{ft}$ (geographic-only alternative).

**Overlap warning for bond issuers.** Bond issuers are large and operate in many states, so a
large share of them are treated by TX (2007), and once treated they leave later cohorts. For NY
(2015) the pool of clean controls among bond issuers may be very small. Check this first (the
script prints treated and control counts and the propensity overlap for each cohort). If overlap
is poor:
1. Make the **continuous** exposure DML the headline result. The dose still varies when almost every firm is touched.
2. Use a threshold design: treated = exposure ≥ 5%, controls = exposure < 1% (the paper's
   geographic-exposure robustness test, Table 9 Panel D).
3. Report cohort-specific ATTs and pool only cohorts that pass an overlap check.

---

## 5. Outcomes

**Main:** $\log(\text{Spread}_{ft})$, where Spread is the amount-weighted annual average bond
spread over the duration-matched Treasury (`firm_spread_bps`, already built from TRACE + FISD).
The log scale makes effects proportional, which matters given a 16–3,400 bps range.

**Alternatives:**
1. Spread in bps, winsorized 1/99.
2. Bond-level spreads (`bond_year_credit_spreads.dta`) in bond-level long differences, with
   features for remaining maturity, coupon, issue size, age, and trading volume. SEs clustered by firm.
3. Offering-yield spreads on new issues (FISD primary market).
4. Credit ratings (FISD ratings table) and CDS spreads (Markit), if available.

**Mechanism outcomes** (the same DML machinery):
1. ROA volatility $t$ to $t+2$, the paper's outcome. This replicates the paper and gives the
   first stage of the risk-taking channel.
2. Equity volatility and Merton distance to default.
3. Book leverage.
4. State ETR.

---

## 6. Controls: pre-determined only

The paper includes contemporaneous controls such as state ETR, leverage, and size. Under combined
reporting, several of these are **outcomes of the treatment**. State ETR changes mechanically, and
leverage responds to tax-shield value (Heider & Ljungqvist 2015). Including them post-treatment
blocks part of the effect. **Main spec:** every element of $X$ is a pre-period (window years
$e-5$ to $e-1$) average. The contemporaneous-control version is a robustness check that matches the paper.

$X_{fc}$ (pre-period means):
- **Firm:** log assets, age, MTB, book leverage, cash surplus, NOL indicator, sales growth, annual
  stock return, ROA, ROA volatility, R&D/assets, number of states.
- **Credit-specific:** average remaining maturity, number of bonds, issue size, trading volume,
  rating (IG/HY), equity volatility.
- **Nexus-weighted state policy:** top tax rate, addback, investment credit, jobs credit, NOL
  carryforward and carryback years, sales-factor weight, throwback, market-based sourcing;
  HQ-state R&D credit.
- **Nexus-weighted economy:** state GDP growth, unemployment rate.
- **Exposure:** pre-existing (pre-1990) combined reporting share; state ETR (pre-period).
- **Fixed-effect analogues:** 2-digit NAICS and cohort indicators. The nuisance learners can
  interact them, which is the role industry×year×cohort FE play in the paper.
- **Optional (robustness):** pre-period mean of the outcome. Conditioning on pre-period spread
  levels guards against mean reversion, but it changes the identifying assumption.

---

## 7. Estimators

Index stacked units by $i=(f,c)$. Work in **long differences** within a firm×cohort. This removes
$\alpha_{cf}$ exactly:

$$\Delta Y_i = \bar Y_{i,\text{post}} - \bar Y_{i,\text{pre}}, \qquad \text{post}=[e, e+4],\ \text{pre}=[e-5, e-1].$$

### 7.1 Binary treatment: DML doubly-robust DiD (DML version of eq. 5)

Assumption: conditional parallel trends, $E[\Delta Y(0)\mid X, D=1] = E[\Delta Y(0)\mid X, D=0]$, plus overlap.

Nuisances, cross-fit over K folds split by firm:
- $m(X) = P(D=1\mid X)$, the propensity score. This replaces entropy balancing.
- $\ell_0(X) = E[\Delta Y\mid X, D=0]$, the counterfactual trend. This replaces $\alpha_{crt}+\lambda C$.

Orthogonal ATT score (Chang 2020):

$$\hat\theta_{ATT} = \frac{\sum_i \big[D_i - (1-D_i)\frac{\hat m(X_i)}{1-\hat m(X_i)}\big]\big(\Delta Y_i - \hat\ell_0(X_i)\big)}{\sum_i D_i}.$$

Influence function: $\phi_i = [D_i(\Delta Y_i-\hat\ell_i) - (1-D_i)\hat w_i(\Delta Y_i-\hat\ell_i) - D_i\hat\theta]/\hat p$.
SE: $\sqrt{\sum_g(\sum_{i\in g}\phi_i)^2}/N$, clustered by firm $g$.

The estimate is consistent if *either* $m$ or $\ell_0$ is estimated well. Errors in each enter
only as a product, which is what allows flexible ML learners while keeping $\sqrt N$ inference.

### 7.2 Event study (DML version of eq. 6)

For each $k\in\{-5,\dots,+4\}\setminus\{-1\}$, apply 7.1 to $\Delta Y_{ik} = Y_{i,e+k}-Y_{i,e-1}$
with the same pre-determined $X$. The $k\le -2$ estimates are placebo and pre-trend tests.

### 7.3 Continuous treatment: DML partially linear model (DML version of eq. 7)

$$\Delta Y_i = \theta\,\Delta T_i + g(X_i) + \varepsilon_i,\qquad \Delta T_i = m_T(X_i) + v_i,$$

where $T$ = CombinedApptTaxRate, so $\Delta T_i$ is the post-period mean exposure for treated
units and 0 for controls. Cross-fit $\ell(X)=E[\Delta Y\mid X]$ and $m_T(X)=E[\Delta T\mid X]$, then

$$\hat\theta = \frac{\sum_i(\Delta Y_i-\hat\ell_i)(\Delta T_i-\hat m_{T,i})}{\sum_i(\Delta T_i-\hat m_{T,i})^2}.$$

$X$ includes the nexus-weighted state tax rate and pre-period state ETR. As in the paper, the
remaining variation in $T$ is then incremental exposure to combined reporting, not exposure to
high-rate states.

Interpretation: $\theta$ is a variance-weighted average of dose-response slopes. Reading it as an
average causal response requires the *strong* parallel-trends assumption in Callaway,
Goodman-Bacon & Sant'Anna (2024). As a check, report DR-DiD ATTs by exposure tercile against the
same controls.

### 7.4 Heterogeneity (the paper's cross-sectional tests)

Split treated firms by a pre-period moderator $Z$ and keep all controls. Estimate 7.1 in each
split. Both estimates use the same control units, so test the difference with the firm-clustered
influence functions: $\text{Var}(\hat\theta_1-\hat\theta_0)$ from $\sum_g(\sum_{i\in g}(\phi^1_i-\phi^0_i))^2/N^2$.

Credit-specific moderators:
- Pre-period NOL/loss position. The insurance channel should be strongest here.
- IG versus HY. The put option is most sensitive near default.
- Pre-period ROA volatility.
- State tax increase versus decrease after adoption. This is the paper's MTR test. It conditions
  on a post-treatment outcome, so it is descriptive only.
- Pre-period patent intensity, for income shifters.
- Number of affiliates/states.

### 7.5 Benchmarks reported alongside

- The paper's stacked OLS DiD with firm×cohort and industry×year×cohort FE, run with `reghdfe` on
  the exported stacked firm-year panel. Entropy-balanced and unbalanced versions.
- Long-difference OLS of $\Delta Y$ on $D$ + industry×cohort dummies + linear $X$ (the 2-period
  equivalent, computed in the script).

Large gaps between DML and OLS point to nonlinear confounding. Spreads are convex in leverage and
volatility, so this is likely.

---

## 8. Implementation choices

- **Cross-fitting:** K = 5 folds, split by **firm**. This keeps the same firm's stacked copies in
  one fold and stops leakage across cohorts.
- **Repeated cross-fitting:** S = 10 fold splits. Report the median $\hat\theta$ and the
  median-adjusted variance $\text{median}_s\{\hat\sigma_s^2 + (\hat\theta_s-\tilde\theta)^2\}$
  (Chernozhukov et al. 2018).
- **Learners:** gradient boosting (main), random forest, and lasso/logit-L1 on standardized
  features. Report all three. Stability across learners is the robustness check.
- **Propensity trimming:** clip at [0.01, 0.99]. Report the share of units near the bounds and
  re-estimate on the overlap sample (0.05–0.95).
- **Nuisance quality:** out-of-fold R² for $\ell$ and $m_T$, out-of-fold log-loss/AUC for $m$.
- **Balance check:** standardized mean differences of $X$ between treated units and controls
  weighted by $\hat m/(1-\hat m)$. This is the DML analogue of the paper's Table 2 Panel C.
- **Weights:** stacked controls appear once per cohort. Weighting treated firms equally, as
  stacking does, is the default. Cohort-by-cohort ATTs aggregated by the number of treated firms
  are reported as well.

---

## 9. Robustness (mirrors the paper's §6.4)

1. Effective-date timing instead of enactment timing.
2. Drop TX, MI, NY (concurrent tax-base changes).
3. Drop states that had addback statutes before adoption.
4. Control for pre-1990 combined reporting exposure, and add nexus-weighted controls computed over combined reporting states only.
5. Geographic-only treatment (CombinedAppt) and threshold treatment (≥5% versus <1%).
6. Contemporaneous controls (the paper's spec), to show what the "bad control" problem does.
7. Sample from 2005 (full TRACE); bond-level outcome; bps outcome.
8. Alternative staggered estimators on the OLS side: Gardner (2021) `did2s`, Wooldridge (2025) ETWFE.
9. DML omitted-variable-bias sensitivity bounds: how strong an unobserved confounder must be to move θ to 0.
10. Placebo: randomly reassign adoption years across never-adopting states (permutation distribution of θ).

---

## 10. Threats specific to credit spreads

- **Macro and credit-cycle confounding.** The 2008–09 cohorts (MA, WV, WI, MI) sit in the
  financial crisis. Within-cohort comparisons difference out common shocks, but treated firms
  differ systematically in size and rating, which correlate with crisis spread behavior.
  Mitigations: rating and size in X, pre-period spread level as a feature (robustness), and
  results reported with and without the 2008/09 cohorts.
- **Anticipation.** Adoption debates often run for years (MD, PA, FL proposals). Use enactment
  timing and inspect k = −2 and −3.
- **Selection into bond markets.** Treatment could change the decision to issue. Report the
  extensive margin (probability of having a TRACE spread) as an outcome.
- **Firm-level spread aggregation.** Composition changes in a firm's bond portfolio move the
  firm-year spread. Use the bond-level outcome with bond-level long differences.

---

## 11. Data inventory

**Already in the project:**
- `empirical/temp/gvkey_year_credit_spreads.dta`: gvkey-year spreads, 2002–2025, 13,814 obs. Use
  this file, not the segment-restricted `..._matvos_with_spreads` file, which keeps only about half of it.
- `empirical/temp/bond_year_credit_spreads.dta`: 97,960 bond-years. Also issuer-week spreads.
- FISD bond characteristics, CCM link, CRSP-bond link, Treasury curves.
- `empirical/temp/cleaned_Compustat.dta`: at, leverage, q, cf, age, sic.

**Still needed (in order of importance):**
1. **Firm × state × year activity shares.** This is the binding constraint. **Data Axle is not
   available** (no WRDS subscription to infogroup_business_academic, checked 2026-11-09). Plan:
   - **Primary: 10-K state exposure from EDGAR** (Garcia & Norli 2012 counts, refined the way the
     paper's OA.8 robustness test does it). Parse each 10-K (cover page and Exhibit 21 excluded).
     Count state names that appear within a 25-word window of operations words (facility, plant,
     store, office, warehouse, employees, operations, located, lease, ...; Hoberg & Moon 2017).
     The firm-year share of state *s* is its count over the total count. Drop state-of-incorporation
     mentions (Delaware boilerplate). Also produce an Item 2 (Properties)-only version.
   - Free data checks: Garcia–Norli posted state counts (check year coverage before relying on it);
     Dyreng & Lindsey Exhibit 21 subsidiary locations (mostly incorporation state, so a robustness check only).
   - Single-state fallback: HQ state from 10-K headers (Bai, Fairhurst & Serfling augmented
     header data) or `comp.company.state`. Use this only for a sanity check.
   - The 10-K share measures *where the firm talks about operating*, not apportionment factors.
     Expect attenuation in the continuous-dose estimate. The binary "any presence" treatment is
     less affected.
   - If an institution can license them later: Data Axle Historical Business, or NETS.
2. **State × year tax-policy panel, 2001–2025:** top corporate rate, combined reporting dates
   (table above), addback, throwback, sales weight, MBS, NOL carryforward/carryback, investment
   credit, jobs credit, R&D credit. Sources: CCH AnswerConnect, Suárez Serrato & Zidar (2016)
   state tax database (to be extended past its end year), Tax Foundation / FTA rate tables.
3. **State GDP (BEA SAGDP) and unemployment (BLS LAUS).**
4. **More Compustat items:** naics, txs, pi, pifo, tlcf, oancf, dpc, xrd, sale, prcc_f, csho, ceq,
   dltt; quarterly oiadpq/atq for ROA volatility. **CRSP** monthly returns (stock return, equity volatility).
5. **Bond ratings** (FISD ratings table).

---

## 12. Code

- `cr_events.csv`: event dates (table above).
- `dml_cr.py`: sample construction (nexus weighting, stacked cohorts, clean controls, long
  differences) and estimators (DR-DiD ATT, event study, PLR, subgroup differences, OLS benchmark,
  diagnostics). Uses only numpy/pandas/scikit-learn.
- `edgar_state_exposure.py`: builds the firm × state × fiscal-year exposure panel from EDGAR
  10-Ks, replacing Data Axle. For each firm-year it uses the first of these that is available:
  1. A unit-count table ("State N" store/branch/facility counts). This is closest to establishment shares.
  2. State mentions near operations words.
  3. All state mentions.
  4. HQ state.

  It reads the main 10-K plus EX-13 (the annual report), skips the cover page, and masks legal and
  governing-law phrases, NYSE / Bank of New York boilerplate, and the filer's own name. It also
  writes an Item 2 (Properties)-only share. Checked on Walmart, Home Depot, Exxon, 3M and TI for
  fiscal 2004 and 2013–15. Example: Home Depot CA = 232/1,977 stores = 0.118.

  Run (1,457 spread firms → roughly 20k filings, about 2–3 hours of downloading at SEC's rate limit):
  ```bash
  export SEC_USER_AGENT="Your Name your.email@school.edu"
  python3 edgar_state_exposure.py index  --years 2000 2025
  python3 edgar_state_exposure.py fetch  --link gvkey_cik.csv --years 2000 2025
  python3 edgar_state_exposure.py parse
  python3 edgar_state_exposure.py shares --link gvkey_cik.csv     # -> exposure_10k.csv
  ```
  `gvkey_cik.csv` comes from WRDS (`select gvkey, cik, conm, state, incorp from comp.company`).
  The output columns `firm, year, state, share` feed directly into `dml_cr.build_stack`.
- `test_synthetic.py`: simulates a firm × state × year world with known effects and nonlinear
  confounding, then checks that the estimators recover the effects.

Input schemas (the `build_*` functions take these as pandas DataFrames):

| Table | Columns |
|---|---|
| `panel` | `firm, year, y, industry` + firm controls |
| `exposure` | `firm, year, state, share` (shares sum to 1 per firm-year) |
| `policy` | `state, year, tax_rate` + other state policy/economy columns |
| `events` | `state, enacted, effective` (years) |

Stata alternative: `ddml interactive … , atet` on the long-difference file gives the same DR-DiD
ATT, and `ddml partial` gives the PLR (Ahrens, Hansen, Schaffer & Wiemann 2024; needs `ssc install ddml pystacked`).
