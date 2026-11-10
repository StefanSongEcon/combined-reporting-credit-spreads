"""
build_state_panel.py -- state x year policy / economy panel, 2000-2025.

  tax_rate   top statutory corporate income tax rate, in percent
             (ITEP State Tax Rate Historical Data, v2026.2, sheet CIT_history)
  unemp      annual average unemployment rate, percent (FRED <ST>UR, BLS LAUS)
  gdp_growth real GDP growth, percent (FRED <ST>RGSP, BEA)
  pre1990_cr indicator: combined reporting state since before 1990

Output: state_panel.csv
"""
import io
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from edgar_state_exposure import STATE_NAMES
from dml_cr import PRE1990_CR_STATES

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw_state"
RAW.mkdir(exist_ok=True)
ITEP_URL = "https://itep.org/wp-content/uploads/2026.2_ITEP-State-Tax-Rate-Historical-Data.xlsx"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
STATES = list(STATE_NAMES) + ["DC"]
NAME2ABBR = {v: k for k, v in STATE_NAMES.items()} | {"District of Columbia": "DC"}


def itep_rates() -> pd.DataFrame:
    f = RAW / "itep_2026.2.xlsx"
    if not f.exists():
        f.write_bytes(requests.get(ITEP_URL, headers=UA, timeout=120).content)
    # the workbook's stylesheet has font-family codes openpyxl rejects; strip them
    zin = zipfile.ZipFile(f)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for it in zin.infolist():
            b = zin.read(it.filename)
            if it.filename.endswith("styles.xml"):
                b = re.sub(rb'<family val="\d+"\s*/>', b"", b)
            zout.writestr(it, b)
    buf.seek(0)
    d = pd.read_excel(buf, sheet_name="CIT_history", header=0)
    d = d.rename(columns={d.columns[0]: "name"})
    d = d[d.name.isin(NAME2ABBR)]
    long = d.melt(id_vars="name", var_name="year", value_name="tax_rate")
    long["year"] = long.year.astype(int)
    long["state"] = long.name.map(NAME2ABBR)
    long["tax_rate"] = pd.to_numeric(long.tax_rate, errors="coerce")
    # ITEP stores rates as fractions (0.065); convert to percent
    if long.tax_rate.max() < 1:
        long["tax_rate"] *= 100
    return long[["state", "year", "tax_rate"]]


def fred(series: str) -> pd.DataFrame:
    f = RAW / f"{series}.csv"
    if not f.exists():
        for i in range(5):
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}",
                             headers=UA, timeout=60)
            if r.status_code == 200:
                f.write_bytes(r.content)
                break
            time.sleep(2 ** i)
    d = pd.read_csv(f)
    d.columns = ["date", "value"]
    d["year"] = pd.to_datetime(d.date).dt.year
    d["value"] = pd.to_numeric(d.value, errors="coerce")
    return d


# ITEP codes non-income business taxes as 0. Texas (franchise/margin tax) and Michigan's
# Single Business Tax are treated as income taxes under ASC 740 and in Welsch (2025), so
# fill them with their statutory rates. Ohio CAT, Washington B&O, etc. stay at 0.
OVERRIDES = {
    "TX": {**{y: 4.5 for y in range(1995, 2008)},            # earned-surplus franchise tax
           **{y: 1.0 for y in range(2008, 2014)}, 2014: 0.975, 2015: 0.95,
           **{y: 0.75 for y in range(2016, 2026)}},          # margin tax
    "MI": {2000: 2.2, 2001: 2.1, 2002: 2.0,
           **{y: 1.9 for y in range(2003, 2008)}},           # Single Business Tax
}


def main():
    rates = itep_rates()
    for s, d in OVERRIDES.items():
        for y, v in d.items():
            m = (rates.state == s) & (rates.year == y)
            if m.any() and rates.loc[m, "tax_rate"].iloc[0] == 0:
                rates.loc[m, "tax_rate"] = v
    rows = []
    for s in STATES:
        u = fred(f"{s}UR").groupby("year").value.mean().rename("unemp")
        g = fred(f"{s}RGSP").groupby("year").value.mean()
        gg = (100 * g.pct_change()).rename("gdp_growth")
        df = pd.concat([u, gg], axis=1).reset_index().rename(columns={"index": "year"})
        df["state"] = s
        rows.append(df)
    econ = pd.concat(rows)
    panel = rates.merge(econ, on=["state", "year"], how="outer")
    panel = panel[panel.year.between(1995, 2025)]
    panel["pre1990_cr"] = panel.state.isin(PRE1990_CR_STATES).astype(int)
    panel = panel.sort_values(["state", "year"])
    panel.to_csv(HERE / "state_panel.csv", index=False)
    chk = panel[panel.year.between(2000, 2025)]
    print(chk.groupby("year")[["tax_rate", "unemp", "gdp_growth"]].mean().round(2).to_string())
    print("missing by column (2000-2025):", chk[["tax_rate", "unemp", "gdp_growth"]].isna().sum().to_dict())
    print(chk[chk.state.isin(["TX", "OH", "NV", "WA", "SD", "WY", "MI", "NY", "VT"]) & chk.year.isin([2005, 2010, 2020])]
          .pivot(index="state", columns="year", values="tax_rate").to_string())


if __name__ == "__main__":
    main()
