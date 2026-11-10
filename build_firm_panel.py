"""
build_firm_panel.py -- gvkey x year panel: TRACE credit spreads + Compustat controls.

Spread year = calendar year of trades (empirical/temp/gvkey_year_credit_spreads.dta).
Compustat fiscal year t is matched to spread year t.

Output: firm_panel.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def winsor(s: pd.Series, p=0.01) -> pd.Series:
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


def main():
    sp = pd.read_stata(ROOT / "empirical/temp/gvkey_year_credit_spreads.dta")
    sp["gvkey"] = sp.gvkey.astype(str).str.zfill(6)
    sp = sp.rename(columns={"year": "fyear"})
    sp["fyear"] = sp.fyear.astype(int)

    c = pd.read_csv(ROOT / "cash_holdings/raw/CompuStat.csv",
                    dtype={"gvkey": str, "naics": str, "sic": str, "cusip": str})
    c = c[c.fyear >= 1990].sort_values(["gvkey", "fyear", "datadate"])
    c = c.drop_duplicates(["gvkey", "fyear"], keep="last")
    first = c.groupby("gvkey").fyear.transform("min")
    allc = pd.read_csv(ROOT / "empirical/raw/Compustat.csv", usecols=["gvkey", "fyear"],
                       dtype={"gvkey": str})
    first_any = allc.groupby("gvkey").fyear.min().rename("first_fyear")
    c = c.merge(first_any, on="gvkey", how="left")
    c["first_fyear"] = c.first_fyear.fillna(first)

    c = c.sort_values(["gvkey", "fyear"])
    g = c.groupby("gvkey")
    at = c["at"].where(c["at"] > 0)
    c["size"] = np.log(at)
    c["book_lev"] = (c.dltt.fillna(0) + c.dlc.fillna(0)) / at
    c["mve"] = c.prcc_f * c.csho
    c["mtb"] = c.mve / c.ceq.where(c.ceq > 0)
    c["mkt_lev"] = (c.dltt.fillna(0) + c.dlc.fillna(0)) / (c.dltt.fillna(0) + c.dlc.fillna(0) + c.mve)
    c["roa"] = c.oibdp / at
    c["cash_at"] = c.che / at
    c["rd_at"] = c.xrd.fillna(0) / at
    c["rd_miss"] = c.xrd.isna().astype(int)
    c["capx_at"] = c.capx / at
    c["tang"] = c.ppent / at
    c["intan_at"] = c.intan / at
    c["loss"] = (c.ib < 0).astype(float).where(c.ib.notna())
    lag_sale = g.sale.shift(1).where(g.fyear.shift(1) == c.fyear - 1)
    c["sales_growth"] = (c.sale - lag_sale) / lag_sale.where(lag_sale > 0)
    c["age"] = c.fyear - c.first_fyear + 1
    # annual ROA volatility over t-4..t (min 3 obs): accounting risk proxy
    c["roa_vol"] = g.roa.transform(lambda s: s.rolling(5, min_periods=3).std())
    c["naics2"] = c.naics.str[:2]
    c["sic2"] = c.sic.str[:2]
    c["industry"] = np.where(c.naics2.notna(), "N" + c.naics2.fillna(""), "S" + c.sic2.fillna(""))

    keep = ["gvkey", "fyear", "industry", "naics2", "sic2", "size", "book_lev", "mkt_lev", "mtb",
            "roa", "cash_at", "rd_at", "rd_miss", "capx_at", "tang", "intan_at", "loss",
            "sales_growth", "age", "roa_vol"]
    c = c[keep]

    p = sp.merge(c, on=["gvkey", "fyear"], how="left")
    # sample filters (Welsch 2025): drop financials, utilities, public sector
    fin = p.naics2.eq("52") | (p.naics2.isna() & p.sic2.between("60", "69"))
    utl = p.naics2.eq("22") | (p.naics2.isna() & p.sic2.eq("49"))
    pub = p.naics2.eq("92") | (p.naics2.isna() & p.sic2.between("91", "99"))
    print(f"firm-years with spreads: {len(p):,}; dropping fin {fin.sum()}, util {utl.sum()}, "
          f"public {pub.sum()}, no Compustat match {p.industry.isna().sum()}")
    p = p[~(fin | utl | pub) & p.industry.notna()].copy()

    p["spread_bps"] = winsor(p.firm_spread_bps)
    p["y"] = np.log(p.firm_spread_bps)              # main outcome: log spread
    p["log_n_bonds"] = np.log(p.n_bonds)
    p["log_volume"] = np.log(p.gvkey_annual_trade_volume)
    p["maturity"] = p.avg_remaining_maturity_yrs
    for v in ["book_lev", "mkt_lev", "mtb", "roa", "cash_at", "rd_at", "capx_at", "tang",
              "intan_at", "sales_growth", "roa_vol", "maturity", "log_volume"]:
        p[v] = winsor(p[v])
    p = p.rename(columns={"gvkey": "firm", "fyear": "year"})
    cols = ["firm", "year", "y", "spread_bps", "industry", "size", "book_lev", "mkt_lev", "mtb",
            "roa", "cash_at", "rd_at", "rd_miss", "capx_at", "tang", "intan_at", "loss",
            "sales_growth", "age", "roa_vol", "maturity", "log_n_bonds", "log_volume"]
    p[cols].to_csv(HERE / "firm_panel.csv", index=False)
    print(f"firm_panel.csv: {len(p):,} firm-years, {p.firm.nunique():,} firms, "
          f"{p.year.min()}-{p.year.max()}")
    print(p[cols[2:]].describe().T[["count", "mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
