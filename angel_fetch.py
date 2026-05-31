"""
Angel One -> Bank Nifty option-chain Parquet fetcher
====================================================
Pulls per-minute CE/PE premium history from Angel One SmartAPI and writes a Parquet
file in the EXACT schema the backtests read:
    datetime, expiry, strike, opt_type, open, high, low, close, volume

Pipeline
--------
  1. login (TOTP)                          -> SmartConnect session
  2. download instrument master (1 file)   -> all NFO contracts
  3. find BANKNIFTY expiries in range (read from master — monthly since Nov 2024,
     and the expiry weekday has changed repeatedly, so never assume it)
  4. anchor ATM per expiry from index daily candles -> choose a strike band
  5. for each (strike, CE/PE): getCandleData in <=N-day chunks, rate-limited
  6. assemble + write Parquet

HONEST LIMITS (read these)
--------------------------
* The instrument master is a CURRENT snapshot. Expired contracts drop off it, so
  you often CANNOT fetch deep history for already-expired strikes. This caps how
  far back Angel One alone lets you backtest. Bank Nifty is MONTHLY-only since
  Nov 2024, so each expiry covers ~a month; a few recent months is realistic,
  multi-year usually is not.
* SmartAPI historical is rate-limited and 1-minute candles are capped per request
  (~30 days). We chunk and sleep; if you get throttled, raise SLEEP_SEC.
* The live API calls in this file were NOT run in the build sandbox (no creds /
  no network to Angel). The pure logic (master parsing, strike scaling, chunking,
  schema assembly) IS unit-tested. Run it yourself to validate the network path.

Setup
-----
  pip install smartapi-python pyotp logzero websocket-client requests polars pandas
  export ANGEL_CLIENT=...    ANGEL_PIN=...    ANGEL_TOTP_SECRET=...    ANGEL_APIKEY=...

Run
---
  python angel_fetch.py --weeks 4 --band 6 --out banknifty_chain.parquet
  # then:
  python options_premium_backtest_fast.py --parquet banknifty_chain.parquet
"""

import os
import time
import argparse
from datetime import datetime, timedelta, date

import requests
import pandas as pd

INSTRUMENT_URL = ("https://margincalculator.angelbroking.com/"
                  "OpenAPI_File/files/OpenAPIScripMaster.json")
SLEEP_SEC = 0.45            # between candle calls; raise if throttled
ONEMIN_CHUNK_DAYS = 25      # stay under the ~30-day 1-minute window
STRIKE_STEP = 100           # Bank Nifty — VERIFY


# --------------------------------------------------------------------------- #
#  LOGIN  (swap this out for your dashboard's auth if you prefer)              #
# --------------------------------------------------------------------------- #
def login():
    import pyotp
    from SmartApi import SmartConnect
    api_key = os.environ["ANGEL_APIKEY"]
    client  = os.environ["ANGEL_CLIENT"]
    pin     = os.environ["ANGEL_PIN"]
    totp    = pyotp.TOTP(os.environ["ANGEL_TOTP_SECRET"]).now()
    sc = SmartConnect(api_key=api_key)
    data = sc.generateSession(client, pin, totp)
    if not data.get("status"):
        raise RuntimeError(f"login failed: {data}")
    return sc


# --------------------------------------------------------------------------- #
#  INSTRUMENT MASTER  (pure logic — unit-tested)                              #
# --------------------------------------------------------------------------- #
def download_instruments() -> list:
    return requests.get(INSTRUMENT_URL, timeout=60).json()


def parse_expiry(s: str) -> date:
    """Angel master expiry strings look like '25JAN2024'."""
    return datetime.strptime(s.strip().upper(), "%d%b%Y").date()


def scale_strike(raw) -> int:
    """Angel master stores strike * 100 (e.g. '4800000' -> 48000)."""
    return int(round(float(raw) / 100.0))


def banknifty_options(master: list) -> pd.DataFrame:
    """Filter the master to Bank Nifty index options, normalized."""
    rows = []
    for r in master:
        if (r.get("exch_seg") == "NFO"
                and r.get("name") == "BANKNIFTY"
                and r.get("instrumenttype") == "OPTIDX"):
            sym = r.get("symbol", "")
            ot = "CE" if sym.endswith("CE") else ("PE" if sym.endswith("PE") else None)
            if ot is None:
                continue
            try:
                rows.append({
                    "token": str(r["token"]),
                    "symbol": sym,
                    "expiry": parse_expiry(r["expiry"]),
                    "strike": scale_strike(r["strike"]),
                    "opt_type": ot,
                })
            except (KeyError, ValueError):
                continue
    return pd.DataFrame(rows)


def index_token(master: list) -> str:
    """Token for the BANKNIFTY spot index (exch_seg NSE)."""
    for r in master:
        if (r.get("name") == "BANKNIFTY" and r.get("exch_seg") == "NSE"
                and r.get("instrumenttype") in ("AMXIDX", "", "INDEX")):
            return str(r["token"])
    # fallback: Angel's well-known Bank Nifty index token
    return "99926009"


# --------------------------------------------------------------------------- #
#  DATE / EXPIRY HELPERS  (pure logic — unit-tested)                          #
# --------------------------------------------------------------------------- #
def expiries_in_range(opts: pd.DataFrame, start: date, end: date) -> list:
    """All actual expiries present in the master within [start, end].
    We do NOT assume weekly/monthly or any weekday — Bank Nifty went monthly-only
    after Nov 2024 and its expiry weekday changed several times (Wed->Thu->Mon->Tue).
    The master is the single source of truth; read whatever expiries it lists."""
    return sorted({e for e in opts["expiry"].unique() if start <= e <= end})


def active_window(expiry: date, all_expiries: list, max_lookback_days: int = 45):
    """Trading window where `expiry` is the front contract: from the day after the
    previous listed expiry up to this expiry. Bank Nifty is monthly, so consecutive
    expiries are ~30-36 days apart; the cap (45) only kicks in when a prior expiry
    is missing from the master, to avoid pulling an unbounded span."""
    prev = [e for e in all_expiries if e < expiry]
    if prev:
        start = max(prev) + timedelta(days=1)
    else:
        start = expiry - timedelta(days=max_lookback_days)
    # safety cap only for pathological gaps (missing prior expiry)
    if (expiry - start).days > max_lookback_days:
        start = expiry - timedelta(days=max_lookback_days)
    return start, expiry


def chunk_ranges(start: date, end: date, max_days: int):
    """Yield (from,to) datetime strings 'YYYY-MM-DD HH:MM' for getCandleData."""
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=max_days - 1), end)
        yield (f"{cur:%Y-%m-%d} 09:15", f"{stop:%Y-%m-%d} 15:30")
        cur = stop + timedelta(days=1)


def nearest_strike(level: float, step: int) -> int:
    return int(round(level / step) * step)


# --------------------------------------------------------------------------- #
#  CANDLE FETCH  (network — NOT run in sandbox)                               #
# --------------------------------------------------------------------------- #
def fetch_candles(sc, exchange: str, token: str, start: date, end: date) -> pd.DataFrame:
    """Per-minute candles for one instrument, chunked + rate-limited."""
    frames = []
    for frm, to in chunk_ranges(start, end, ONEMIN_CHUNK_DAYS):
        params = {"exchange": exchange, "symboltoken": token,
                  "interval": "ONE_MINUTE", "fromdate": frm, "todate": to}
        for attempt in range(4):
            try:
                resp = sc.getCandleData(params)
                if resp.get("status") and resp.get("data"):
                    frames.append(pd.DataFrame(
                        resp["data"],
                        columns=["dt", "open", "high", "low", "close", "volume"]))
                break
            except Exception as e:
                wait = SLEEP_SEC * (2 ** attempt)
                print(f"  retry {attempt+1} ({token} {frm[:10]}): {e} -> sleep {wait:.1f}s")
                time.sleep(wait)
        time.sleep(SLEEP_SEC)
    if not frames:
        return pd.DataFrame(columns=["dt", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames, ignore_index=True)
    out["dt"] = pd.to_datetime(out["dt"]).dt.tz_localize(None)
    return out


def daily_index_close(sc, token: str, start: date, end: date) -> dict:
    """date -> close for the index, to anchor ATM each day."""
    out = {}
    for frm, to in chunk_ranges(start, end, 90):
        params = {"exchange": "NSE", "symboltoken": token,
                  "interval": "ONE_DAY", "fromdate": frm, "todate": to}
        try:
            resp = sc.getCandleData(params)
            for row in resp.get("data", []):
                d = pd.to_datetime(row[0]).date()
                out[d] = float(row[4])
        except Exception as e:
            print(f"  index fetch issue: {e}")
        time.sleep(SLEEP_SEC)
    return out


# --------------------------------------------------------------------------- #
#  ASSEMBLE  (pure logic — unit-tested)                                       #
# --------------------------------------------------------------------------- #
def assemble(candle_frames: list) -> pd.DataFrame:
    """Each item: (expiry, strike, opt_type, candles_df). -> backtest schema."""
    pieces = []
    for expiry, strike, ot, c in candle_frames:
        if c.empty:
            continue
        c = c.copy()
        c["expiry"] = expiry
        c["strike"] = strike
        c["opt_type"] = ot
        c.rename(columns={"dt": "datetime"}, inplace=True)
        pieces.append(c[["datetime", "expiry", "strike", "opt_type",
                         "open", "high", "low", "close", "volume"]])
    if not pieces:
        return pd.DataFrame(columns=["datetime", "expiry", "strike", "opt_type",
                                     "open", "high", "low", "close", "volume"])
    return pd.concat(pieces, ignore_index=True).sort_values("datetime")


# --------------------------------------------------------------------------- #
#  ORCHESTRATION                                                              #
# --------------------------------------------------------------------------- #
def build(weeks: int, band: int, out_path: str, all_expiries: bool = False):
    today = date.today()
    # In all-expiries mode: fetch ALL contracts listed in the master, going back
    # as far as each contract has been trading (up to 180 days before expiry).
    # This maximises historical data across multiple monthly cycles.
    if all_expiries:
        data_start = today - timedelta(days=180)
        exp_end = today + timedelta(days=400)  # grab every expiry in the master
    else:
        data_start = today - timedelta(weeks=weeks)
        exp_end = today + timedelta(days=max(weeks * 7, 45))

    print(f"[1] login"); sc = login()
    print(f"[2] instrument master"); master = download_instruments()
    opts = banknifty_options(master)
    idx_tok = index_token(master)
    print(f"    {len(opts)} BANKNIFTY option contracts in master")

    exps = expiries_in_range(opts, data_start, exp_end)
    print(f"[3] {len(exps)} expiries in range (read from master): {exps}")
    if not exps:
        print("    none in range — expired contracts drop off the master.")
        return

    # For index ATM anchoring we need closes from data_start to today
    print(f"[4] index daily closes to anchor ATM")
    idx_close = daily_index_close(sc, idx_tok, data_start, today)

    jobs = []   # (exchange, token, expiry, strike, opt_type, win_start, win_end)
    for exp in exps:
        if all_expiries:
            # Each contract's window: 180 days before expiry up to today (not future)
            w0 = max(exp - timedelta(days=180), data_start)
            w1 = min(exp, today)
        else:
            w0, w1 = active_window(exp, exps)
            w1 = min(w1, today)
        # ATM reference = average index close over the window
        refs = [idx_close[d] for d in idx_close if w0 <= d <= w1]
        if not refs:
            print(f"    skipping {exp} — no index closes in window {w0}..{w1}")
            continue
        atm = nearest_strike(sum(refs) / len(refs), STRIKE_STEP)
        wanted = {atm + k * STRIKE_STEP for k in range(-band, band + 1)}
        sub = opts[(opts["expiry"] == exp) & (opts["strike"].isin(wanted))]
        for r in sub.itertuples():
            jobs.append(("NFO", r.token, exp, r.strike, r.opt_type, w0, w1))

    print(f"[5] fetching {len(jobs)} contracts (~{len(jobs)} calls min, rate-limited)")
    frames = []
    for i, (exch, tok, exp, strike, ot, w0, w1) in enumerate(jobs, 1):
        print(f"    [{i}/{len(jobs)}] {exp} {strike}{ot}")
        c = fetch_candles(sc, exch, tok, w0, w1)
        frames.append((exp, strike, ot, c))

    print(f"[6] assemble + write {out_path}")
    df = assemble(frames)
    if df.empty:
        print("    no data assembled — check limits above."); return
    try:
        df.to_parquet(out_path, index=False)          # pandas handles dtypes cleanly
    except Exception as e:
        print(f"    parquet engine missing (pip install pyarrow): {e}")
        df.to_csv(out_path.replace('.parquet', '.csv'), index=False)
        print("    fell back to CSV")
    print(f"    wrote {len(df):,} rows, {df['datetime'].dt.date.nunique()} sessions -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=10,
                    help="how far back from today (Bank Nifty is monthly, so >=8 "
                         "to catch a full expiry window)")
    ap.add_argument("--band", type=int, default=6, help="strikes each side of ATM")
    ap.add_argument("--out", default="banknifty_chain.parquet")
    ap.add_argument("--all-expiries", action="store_true",
                    help="fetch ALL expiries in the master with max historical depth")
    args = ap.parse_args()
    build(args.weeks, args.band, args.out, all_expiries=args.all_expiries)


if __name__ == "__main__":
    main()