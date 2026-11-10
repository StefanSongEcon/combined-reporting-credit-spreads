"""
edgar_state_exposure.py -- firm x state x fiscal-year exposure shares from 10-K text.

Replaces Data Axle establishment shares (Welsch 2025) with state mentions in 10-K
filings (Garcia & Norli 2012), restricted to operations context (Hoberg & Moon 2017
style window; the paper's OA.8 alternative treatment).

Stages (each is resumable; downloads are cached under --cache):

  index   download EDGAR quarterly form indexes -> 10-K filing list
  fetch   for the CIKs in --link (or --ciks): filing index page (period of report,
          HQ state, state of incorporation, main document) + main 10-K document,
          stored as gzipped plain text
  parse   count state mentions per filing:
            n_all   all mentions in the body (cover page dropped)
            n_ops   mentions within +/-25 words of an operations word
            n_item2 mentions inside Item 2 (Properties)
          legal-context mentions (incorporated / laws of / chancery ...) and the
          filer's own name (e.g. "Texas Instruments") are masked first
            n_units units in 'State  N' tables (store/branch/facility counts)
  shares  firm-year shares, first available of: unit-count table (n_units), operations-
          context mentions (n_ops), all mentions (n_all), HQ state = 1. Writes exposure.csv (firm=gvkey, year=fyear, state, share, source)

SEC fair-access policy: identify yourself. Set
    export SEC_USER_AGENT="Your Name your.email@school.edu"
The script keeps requests under 8 per second.

Example
    python3 edgar_state_exposure.py index  --years 2000 2025
    python3 edgar_state_exposure.py fetch  --link gvkey_cik.csv
    python3 edgar_state_exposure.py parse
    python3 edgar_state_exposure.py shares --link gvkey_cik.csv
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import html
import os
import re
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
SEC = "https://www.sec.gov"
FORMS = {"10-K", "10-K405", "10-KT"}  # amendments (10-K/A) excluded

# ---------------------------------------------------------------------------
# State dictionary (full names only; abbreviations are too noisy in prose)
# ---------------------------------------------------------------------------
STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
# DC handled separately (before Washington)
DC_PAT = re.compile(r"\b(?:washington,?\s*d\.?\s*c\.?|district\s+of\s+columbia)", re.I)

# phrases that contain a state name but say nothing about operations
NOISE_PHRASES = [
    r"new\s+york\s+stock\s+exchange", r"bank\s+of\s+new\s+york(?:\s+mellon)?",
    r"federal\s+reserve\s+bank\s+of\s+\w+(?:\s+\w+)?", r"new\s+york\s+mellon",
    r"new\s+york\s+times", r"george\s+washington", r"washington\s+mutual",
    r"wall\s+street\s+journal", r"kansas\s+city\s+southern",
    r"(?:northern|southern|eastern|western|central|middle)\s+district\s+of\s+[a-z]+(?:\s+[a-z]+)?",
    r"district\s+court\s+for\s+the\s+[a-z]+\s+district\s+of\s+[a-z]+(?:\s+[a-z]+)?",
    r"supreme\s+court\s+of\s+(?:the\s+state\s+of\s+)?[a-z]+(?:\s+[a-z]+)?",
]
NOISE_PAT = re.compile("|".join(NOISE_PHRASES), re.I)
# legal / governing-law context: drop a state mention if one of these is within 8 words
LEGAL_PAT = re.compile(r"incorporat|organi[sz]ed\s+under|corporation\s+law|chancery|"
                       r"laws\s+of|formed\s+under|domiciled|statutory\s+trust|"
                       r"limited\s+liability\s+company\s+organized|governed\s+by", re.I)
OPS_WORDS = re.compile(
    r"^(?:facilit\w*|plants?|factor(?:y|ies)|stores?|offices?|warehouses?|distribution|"
    r"headquarter\w*|located|locations?|operat\w*|employ\w*|leased?|leases|manufactur\w*|"
    r"mills?|mines?|refiner(?:y|ies)|terminals?|branch\w*|restaurants?|hotels?|hospitals?|"
    r"campus\w*|sites?|propert(?:y|ies)|acres?|square|production|centers?|pipelines?|wells?|"
    r"stations?|outlets?|depots?|yards?|casinos?|clinics?|schools?|dealerships?|"
    r"customers?|markets?|sales|service\s*area)$", re.I)
WINDOW = 25


def _state_regex():
    # longest names first so "West Virginia" is consumed before "Virginia", etc.
    items = sorted(STATE_NAMES.items(), key=lambda kv: -len(kv[1]))
    alt = "|".join(re.escape(n).replace(r"\ ", r"\s+") for _, n in items)
    return re.compile(rf"\b({alt})\b", re.I), {n.lower(): s for s, n in STATE_NAMES.items()}


STATE_PAT, NAME2ABBR = _state_regex()


# ---------------------------------------------------------------------------
# HTTP with a global rate limit
# ---------------------------------------------------------------------------
class SecClient:
    def __init__(self, max_rps: float = 8.0):
        ua = os.environ.get("SEC_USER_AGENT")
        if not ua:
            sys.exit("Set SEC_USER_AGENT='Your Name your.email@school.edu' (SEC fair-access policy).")
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self.min_gap = 1.0 / max_rps
        self.lock = threading.Lock()
        self.last = 0.0

    def get(self, url: str, tries: int = 6) -> bytes | None:
        for i in range(tries):
            with self.lock:
                wait = self.last + self.min_gap - time.time()
                if wait > 0:
                    time.sleep(wait)
                self.last = time.time()
            try:
                r = self.s.get(url, timeout=60)
            except requests.RequestException:
                time.sleep(2 ** i)
                continue
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
            time.sleep(min(60, 2 ** (i + 1)))  # 403/429/5xx: back off
        return None


# ---------------------------------------------------------------------------
# Stage 1: index
# ---------------------------------------------------------------------------
def stage_index(cache: Path, y0: int, y1: int):
    cli = SecClient()
    d = cache / "index"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for y in range(y0, y1 + 1):
        for q in (1, 2, 3, 4):
            # keep only the 10-K lines of each quarterly index (raw files are ~30-50 MB)
            f = d / f"form_{y}Q{q}.10k.txt"
            if not f.exists():
                b = cli.get(f"{SEC}/Archives/edgar/full-index/{y}/QTR{q}/form.idx")
                if b is None:
                    continue
                keep = [ln for ln in b.decode("latin-1").splitlines() if ln[:12].strip() in FORMS]
                f.write_text("\n".join(keep))
            for line in f.read_text(errors="ignore").splitlines():
                form = line[:12].strip()
                if form not in FORMS:
                    continue
                m = re.match(r"(.+?)\s+(\d+)\s+(\d{4}-\d{2}-\d{2})\s+(edgar/\S+)\s*$", line[12:])
                if m:
                    rows.append((form, m.group(1).strip(), int(m.group(2)), m.group(3), m.group(4)))
    idx = pd.DataFrame(rows, columns=["form", "company", "cik", "date_filed", "path"])
    idx["accession"] = idx.path.str.extract(r"/([\d-]+)\.txt$")[0]
    idx.to_csv(cache / "edgar_10k_index.csv", index=False)
    print(f"10-K filings indexed: {len(idx):,}  -> {cache / 'edgar_10k_index.csv'}")


# ---------------------------------------------------------------------------
# Stage 2: fetch
# ---------------------------------------------------------------------------
def html_to_text(raw: bytes) -> str:
    s = raw.decode("utf-8", errors="ignore")
    s = re.sub(r"(?is)<ix:header>.*?</ix:header>", " ", s)          # hidden inline-XBRL facts
    s = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", s)
    s = re.sub(r"(?is)<div[^>]*display:\s*none[^>]*>.*?</div>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = s.replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse_index_page(page: str) -> dict:
    t = html.unescape(re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", page)))
    out = {}
    m = re.search(r"Period of Report\s+(\d{4}-\d{2}-\d{2})", t)
    out["period"] = m.group(1) if m else None
    m = re.search(r"State of Incorp\.?:\s*([A-Z0-9]{2})", t)
    out["inc_state"] = m.group(1) if m else None
    m = re.search(r"Business Address\s+(.*?)(?:Mailing Address|\(Filer\)|$)", t)
    hq = None
    if m:
        mm = re.findall(r"\b([A-Z]{2})\s+\d{5}(?:-\d{4})?\b", m.group(1))
        hq = mm[-1] if mm else None
    out["hq_state"] = hq
    # main document: first table row whose Type is the 10-K form
    # main document = first row whose Type is the 10-K form; also the annual report to
    # shareholders (EX-13), where many filers put state unit counts / property lists
    main, ex13 = None, None
    for row in re.findall(r"(?is)<tr.*?</tr>", page):
        cells = [html.unescape(re.sub(r"(?s)<[^>]+>", "", c)).strip()
                 for c in re.findall(r"(?is)<td.*?</td>", row)]
        href = re.search(r'href="([^"]+)"', row)
        if not href or len(cells) < 4:
            continue
        typ = cells[3].upper()
        if main is None and typ in FORMS:
            main = href.group(1)
        elif ex13 is None and typ.startswith("EX-13") and not href.group(1).lower().endswith((".pdf", ".jpg", ".gif")):
            ex13 = href.group(1)
    strip = lambda u: u[len("/ix?doc="):] if u and u.startswith("/ix?doc=") else u
    out["main_doc"], out["ex13_doc"] = strip(main), strip(ex13)
    return out


def _split_submission(raw: bytes) -> tuple[bytes | None, bytes | None]:
    """Main 10-K and EX-13 documents from a complete submission .txt file."""
    main = ex13 = None
    for doc in re.findall(rb"(?is)<DOCUMENT>(.*?)</DOCUMENT>", raw):
        m = re.search(rb"<TYPE>\s*([^\s<]+)", doc)
        typ = m.group(1).decode("latin-1").upper() if m else ""
        body = re.search(rb"(?is)<TEXT>(.*?)(?:</TEXT>|$)", doc)
        body = body.group(1) if body else doc
        if main is None and typ in FORMS:
            main = body
        elif ex13 is None and typ.startswith("EX-13"):
            ex13 = body
    return main, ex13


def _is_dir(u) -> bool:
    return not isinstance(u, str) or u.endswith("/")


def fetch_one(cli: SecClient, cache: Path, row) -> dict | None:
    acc = row.accession
    acc_nd = acc.replace("-", "")
    tdir = cache / "text" / str(row.cik)
    tdir.mkdir(parents=True, exist_ok=True)
    tfile = tdir / f"{acc}.txt.gz"
    mfile = tdir / f"{acc}.meta.csv"
    if tfile.exists() and mfile.exists():
        old = pd.read_csv(mfile).iloc[0].to_dict()
        if not (_is_dir(old.get("main_doc")) and old.get("source_doc") != "submission_txt"):
            return old
    page = cli.get(f"{SEC}/Archives/edgar/data/{row.cik}/{acc_nd}/{acc}-index.htm")
    if page is None:
        return None
    meta = parse_index_page(page.decode("utf-8", errors="ignore"))
    meta.update(cik=row.cik, accession=acc, form=row.form, company=row.company,
                date_filed=row.date_filed)
    raw = raw13 = None
    if not _is_dir(meta["main_doc"]):
        meta["source_doc"] = "main_doc"
        raw = cli.get(SEC + meta["main_doc"])
        if meta.get("ex13_doc") and not _is_dir(meta["ex13_doc"]):
            raw13 = cli.get(SEC + meta["ex13_doc"])
    if raw is None:
        # older filings: documents have no separate files -> use the full submission
        sub = cli.get(f"{SEC}/Archives/{row.path}")
        if sub is None:
            return None
        raw, raw13 = _split_submission(sub)
        meta["source_doc"] = "submission_txt"
        if raw is None:
            return None
    text = html_to_text(raw)
    with gzip.open(tfile, "wt", encoding="utf-8") as fh:
        fh.write(text)
    meta["n_words"] = len(text.split())
    meta["n_words_ex13"] = 0
    if raw13 is not None:
        t13 = html_to_text(raw13)
        with gzip.open(tdir / f"{acc}.ex13.txt.gz", "wt", encoding="utf-8") as fh:
            fh.write(t13)
        meta["n_words_ex13"] = len(t13.split())
    pd.DataFrame([meta]).to_csv(mfile, index=False)
    return meta


def load_link(path: str) -> pd.DataFrame:
    lk = pd.read_csv(path, dtype={"gvkey": str})
    lk = lk.dropna(subset=["cik"])
    lk["cik"] = lk["cik"].astype(float).astype(int)
    lk["gvkey"] = lk["gvkey"].str.zfill(6)
    return lk


def stage_fetch(cache: Path, link: str | None, ciks: list[int] | None, y0: int, y1: int,
                workers: int = 4):
    idx = pd.read_csv(cache / "edgar_10k_index.csv")
    want = set(ciks or []) | (set(load_link(link).cik) if link else set())
    idx = idx[idx.cik.isin(want)]
    idx = idx[idx.date_filed.str[:4].astype(int).between(y0, y1)]
    print(f"filings to fetch: {len(idx):,} for {idx.cik.nunique():,} CIKs")
    cli = SecClient()
    metas, done, t0 = [], 0, time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch_one, cli, cache, r) for r in idx.itertuples()]
        for f in cf.as_completed(futs):
            m = f.result()
            if m:
                metas.append(m)
            done += 1
            if done % 250 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done:,}/{len(idx):,} filings  ({rate:.1f}/s, "
                      f"eta {(len(idx) - done) / rate / 60:.0f} min)", flush=True)
    meta = pd.DataFrame(metas)
    meta.to_csv(cache / "filings_meta.csv", index=False)
    print(f"fetched {len(meta):,} filings in {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------
# Stage 3: parse
# ---------------------------------------------------------------------------
ITEM1 = re.compile(r"\bitem\s*1\s*[\.\:\-–—]?\s*business\b", re.I)
PART1 = re.compile(r"\bpart\s+i\b(?!i)", re.I)
ITEM2 = re.compile(r"\bitem\s*2\s*[\.\:\-–—]?\s*(?:description\s+of\s+)?propert(?:y|ies)\b", re.I)
ITEM3 = re.compile(r"\bitem\s*3\s*[\.\:\-–—]?\s*legal\b", re.I)
SUFFIX = re.compile(r"\b(inc|corp|corporation|co|company|ltd|llc|lp|plc|holdings?|group|the|"
                    r"incorporated|trust|/[a-z]{2}/?)\b\.?", re.I)


NEXT_ITEM = re.compile(r"\bitem\s*(?:1a|1b|2)\b", re.I)


def body_start(text: str) -> int:
    """Skip the cover page and table of contents.

    The real 'Item 1. Business' heading starts a long section; table-of-contents
    entries and cross-references ('see Item 1. Business above') are followed
    quickly by the next item heading or come after the section itself. Take the
    first match followed by >= 3,000 characters before the next item heading.
    """
    m = list(ITEM1.finditer(text))
    if m:
        gaps = []
        for x in m:
            nxt = NEXT_ITEM.search(text, x.end())
            gaps.append((nxt.start() if nxt else len(text)) - x.end())
        for x, g in zip(m, gaps):
            if g >= 3000:
                return x.start()
        return m[max(range(len(m)), key=gaps.__getitem__)].start()
    p = list(PART1.finditer(text))
    if len(p) >= 2:
        return p[1].start()
    return int(0.03 * len(text))


def item2_span(text: str) -> tuple[int, int] | None:
    """Longest 'Item 2. Properties' ... 'Item 3. Legal' segment (ToC entries are short)."""
    best = None
    for m in ITEM2.finditer(text):
        e = ITEM3.search(text, m.end())
        end = e.start() if e else min(len(text), m.end() + 60000)
        if best is None or end - m.start() > best[1] - best[0]:
            best = (m.start(), end)
    return best


def own_name_pattern(company: str):
    words = [w for w in re.split(r"\s+", SUFFIX.sub(" ", str(company))) if w and w.isalpha()]
    name = " ".join(words[:2])
    if name and STATE_PAT.search(name):
        return re.compile(r"\b" + r"\s+".join(map(re.escape, words[:2])) + r"\b", re.I)
    return None


def _table_regex():
    items = sorted(STATE_NAMES.values(), key=lambda n: -len(n))
    alt = "|".join(re.escape(n).replace(r"\ ", r"\s+") for n in items)
    dc = r"washington,?\s*d\.?\s*c\.?|district\s+of\s+columbia"
    return re.compile(rf"(?P<name>{dc}|\b(?:{alt})\b)\s+"
                      rf"(?P<nums>(?:(?:\d[\d,]*|[\u2014\u2013-])\s+){{0,6}}\d[\d,]*)(?![\d.,%])", re.I)


TABLE_ROW = _table_regex()


def unit_table_counts(body: str, gap: int = 300, min_states: int = 10) -> dict:
    """Units by state from 'State  N' tables (store / branch / facility counts).

    Rows like 'Alabama 28', 'Texas 345 27 50 77 499' (last column = row total) or
    'Texas 80 219 69 28' (format columns, summed).
    Rows closer than `gap` characters form one table; tables with fewer than
    `min_states` distinct states, or whose values look like years, are ignored.
    """
    rows = []
    for m in TABLE_ROW.finditer(body):
        name = m.group("name")
        st = "DC" if DC_PAT.fullmatch(name) else NAME2ABBR[re.sub(r"\s+", " ", name.lower())]
        nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", m.group("nums"))]
        # multi-column row: last column is the row total if it equals the sum of the
        # others (dashes count as 0); otherwise the columns are formats -> add them up
        val = nums[-1] if len(nums) == 1 or nums[-1] == sum(nums[:-1]) else sum(nums)
        rows.append((m.start(), m.end(), st, val))
    out, cluster, last_end = {}, [], -10**9

    def flush(cl):
        if len({r[2] for r in cl}) < min_states:
            return
        vals = [r[3] for r in cl]
        if sum(1990 <= v <= 2035 for v in vals) > 0.5 * len(vals):
            return
        for _, _, st, v in cl:
            out[st] = out.get(st, 0) + v

    for r in rows:
        if r[0] - last_end > gap and cluster:
            flush(cluster)
            cluster = []
        cluster.append(r)
        last_end = r[1]
    if cluster:
        flush(cluster)
    return out


def count_states(text: str, company: str = "", skip_cover: bool = True) -> pd.DataFrame:
    start = body_start(text) if skip_cover else 0
    body = text[start:]
    body = NOISE_PAT.sub(" ", body)
    own = own_name_pattern(company)
    if own:
        body = own.sub(" ", body)
    i2 = item2_span(body) if skip_cover else None
    # tokenise once; map character offsets to word positions
    tokens = [(m.start(), m.group()) for m in re.finditer(r"\S+", body)]
    starts = [t[0] for t in tokens]
    is_ops = [bool(OPS_WORDS.match(re.sub(r"[^A-Za-z]", "", w))) for _, w in tokens]
    # prefix sums for O(1) window queries
    ops_cum = [0]
    for a in is_ops:
        ops_cum.append(ops_cum[-1] + a)
    # multiword legal phrases ("laws of", "organized under") -> mark by char offsets
    legal_pos = [m.start() for m in LEGAL_PAT.finditer(body)]

    import bisect

    def tok_index(pos):
        return max(0, bisect.bisect_right(starts, pos) - 1)

    legal_tok = sorted({tok_index(p) for p in legal_pos})

    def near_legal(k):
        j = bisect.bisect_left(legal_tok, k - 8)
        return j < len(legal_tok) and legal_tok[j] <= k + 8

    hits = []
    for m in DC_PAT.finditer(body):
        hits.append(("DC", m.start()))
    masked = DC_PAT.sub(lambda m: " " * len(m.group()), body)
    for m in STATE_PAT.finditer(masked):
        hits.append((NAME2ABBR[re.sub(r"\s+", " ", m.group(1).lower())], m.start()))

    rec = {}
    for st, pos in hits:
        k = tok_index(pos)
        if near_legal(k):
            continue
        lo, hi = max(0, k - WINDOW), min(len(tokens), k + WINDOW + 1)
        ops = (ops_cum[hi] - ops_cum[lo]) > 0
        in2 = bool(i2 and i2[0] <= pos < i2[1])
        r = rec.setdefault(st, [0, 0, 0])
        r[0] += 1
        r[1] += ops
        r[2] += in2
    units = unit_table_counts(body)
    for st in units:
        rec.setdefault(st, [0, 0, 0])
    return pd.DataFrame([(s, *v, units.get(s, 0)) for s, v in rec.items()],
                        columns=["state", "n_all", "n_ops", "n_item2", "n_units"])


def fyear_from_period(period: str | float) -> float:
    if not isinstance(period, str):
        return float("nan")
    y, m = int(period[:4]), int(period[5:7])
    return y if m >= 6 else y - 1  # Compustat fyear convention


def _parse_one(args):
    cache, cik, acc, company = args
    f = cache / "text" / str(cik) / f"{acc}.txt.gz"
    try:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            c = count_states(fh.read(), company)
        f13 = cache / "text" / str(cik) / f"{acc}.ex13.txt.gz"
        if f13.exists():
            with gzip.open(f13, "rt", encoding="utf-8") as fh:
                c13 = count_states(fh.read(), company, skip_cover=False)
            c = pd.concat([c, c13]).groupby("state", as_index=False).sum()
    except (OSError, EOFError):          # truncated file (e.g. interrupted download)
        return None
    c.insert(0, "accession", acc)
    return c


def stage_parse(cache: Path, workers: int = 6):
    import multiprocessing as mp
    metas = []
    for p in (cache / "text").glob("*/*.meta.csv"):
        try:
            metas.append(pd.read_csv(p))
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            pass
    meta = pd.concat(metas, ignore_index=True)
    meta["fyear"] = meta.period.map(fyear_from_period)
    args = [(cache, r.cik, r.accession, r.company) for r in meta.itertuples()]
    out, t0 = [], time.time()
    with mp.Pool(workers) as pool:
        for i, c in enumerate(pool.imap_unordered(_parse_one, args, chunksize=20)):
            if c is not None:
                out.append(c)
            if (i + 1) % 2000 == 0:
                print(f"  parsed {i + 1:,}/{len(meta):,} ({time.time() - t0:.0f}s)", flush=True)
    counts = pd.concat(out, ignore_index=True)
    counts = counts.merge(meta[["accession", "cik", "fyear", "period", "date_filed", "form",
                                "hq_state", "inc_state", "n_words"]], on="accession")
    meta.to_csv(cache / "filings_meta.csv", index=False)
    counts.to_csv(cache / "state_counts_by_filing.csv", index=False)
    print(f"state counts: {len(counts):,} filing-state rows -> {cache / 'state_counts_by_filing.csv'}")


# ---------------------------------------------------------------------------
# Stage 4: shares
# ---------------------------------------------------------------------------
def stage_shares(cache: Path, link: str | None, out: Path):
    meta = pd.read_csv(cache / "filings_meta.csv")
    counts = pd.read_csv(cache / "state_counts_by_filing.csv")
    meta = meta.dropna(subset=["fyear"])
    # one filing per cik-fyear: the latest filed
    meta = meta.sort_values("date_filed").groupby(["cik", "fyear"]).tail(1)
    counts = counts[counts.accession.isin(meta.accession)]

    rows = []
    for acc, g in counts.groupby("accession"):
        for col, src in (("n_units", "10k_table"), ("n_ops", "10k_ops"), ("n_all", "10k_all")):
            tot = g[col].sum()
            if tot > 0:
                for s, v in zip(g.state, g[col]):
                    if v > 0:
                        rows.append((acc, s, v / tot, src))
                break
    sh = pd.DataFrame(rows, columns=["accession", "state", "share", "source"])
    # HQ fallback for filings with no usable mentions
    miss = meta[~meta.accession.isin(sh.accession) & meta.hq_state.isin(list(STATE_NAMES) + ["DC"])]
    sh = pd.concat([sh, pd.DataFrame({"accession": miss.accession, "state": miss.hq_state,
                                      "share": 1.0, "source": "hq_only"})])
    sh = sh.merge(meta[["accession", "cik", "fyear", "hq_state", "inc_state"]], on="accession")
    # Item-2-only alternative shares
    i2 = counts[counts.n_item2 > 0].copy()
    i2["share_item2"] = i2.n_item2 / i2.groupby("accession").n_item2.transform("sum")
    sh = sh.merge(i2[["accession", "state", "share_item2"]], on=["accession", "state"], how="left")

    if link:
        lk = load_link(link)[["gvkey", "cik"]].drop_duplicates()
        sh = sh.merge(lk, on="cik", how="inner")
        sh = sh.rename(columns={"gvkey": "firm"})
    else:
        sh["firm"] = sh["cik"]
    sh = sh.rename(columns={"fyear": "year"})
    sh["year"] = sh["year"].astype(int)
    cols = ["firm", "year", "state", "share", "share_item2", "source", "cik", "accession",
            "hq_state", "inc_state"]
    sh[cols].sort_values(["firm", "year", "share"], ascending=[True, True, False]) \
        .to_csv(out, index=False)
    fy = sh.groupby(["firm", "year"]).source.first().value_counts()
    print(f"exposure rows {len(sh):,}; firm-years by source:\n{fy.to_string()}\n-> {out}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["index", "fetch", "parse", "shares"])
    ap.add_argument("--cache", default=str(HERE / "edgar_cache"))
    ap.add_argument("--years", nargs=2, type=int, default=[2000, 2025])
    ap.add_argument("--link", help="CSV with gvkey,cik (WRDS comp.company)")
    ap.add_argument("--ciks", nargs="*", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "exposure_10k.csv"))
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    if a.stage == "index":
        stage_index(cache, *a.years)
    elif a.stage == "fetch":
        stage_fetch(cache, a.link, a.ciks, *a.years, workers=a.workers)
    elif a.stage == "parse":
        stage_parse(cache, workers=a.workers if a.workers != 4 else 6)
    else:
        stage_shares(cache, a.link, Path(a.out))


if __name__ == "__main__":
    main()
