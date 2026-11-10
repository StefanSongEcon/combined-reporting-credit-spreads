#!/bin/zsh
# Full pipeline after the first 10-K fetch pass: fallback re-fetch, parse, shares, analysis.
set -e
cd "$(dirname "$0")"
: "${SEC_USER_AGENT:?set SEC_USER_AGENT='Your Name your.email@domain' (SEC fair-access policy)}"
export OMP_NUM_THREADS=4
until grep -q "^EXIT" results/edgar_fetch.log 2>/dev/null; do sleep 30; done
echo "[$(date)] pass 2: re-fetch old filings via submission text"
python3 -u edgar_state_exposure.py fetch --link gvkey_cik_spreadfirms.csv --years 2000 2025 > results/edgar_fetch2.log 2>&1
echo "[$(date)] parse"
python3 -u edgar_state_exposure.py parse > results/edgar_parse.log 2>&1
echo "[$(date)] shares"
python3 -u edgar_state_exposure.py shares --link gvkey_cik.csv --out exposure_10k.csv > results/edgar_shares.log 2>&1
echo "[$(date)] analysis"
python3 -u -W ignore run_analysis.py > results/analysis_stdout.log 2>&1
echo "[$(date)] PIPELINE DONE"
