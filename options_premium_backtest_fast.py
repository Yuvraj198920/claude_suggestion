"""
Intraday Premium-Selling Backtest — FAST build (Parquet + Polars + numba)
=========================================================================

Same strategy and SAME LOGIC as options_premium_backtest.py (the readable pandas
version). The only changes are the substrate, so results should match to the rupee
(any tiny difference would be float-format or an |CE-PE| tie, never the algorithm):

  * IO        : Parquet (dtypes preserved, columnar, no re-parsing)  <- biggest win
  * reshape   : Polars (multi-threaded filtering / joins) instead of pandas
  * hot loop  : numba @njit kernel for the per-minute per-leg stop scan
  * metrics   : still pandas on the ~N-row trade table (tiny; kept identical to v1)

Run
---
  python options_premium_backtest_fast.py                          # synthetic smoke test
  python options_premium_backtest_fast.py --parquet chain.parquet  # your data (fast path)
  python options_premium_backtest_fast.py --csv chain.csv          # converts to parquet once

Compare against the pandas version on the SAME data; the numbers should line up and
this one should be markedly faster, especially on multi-year minute data.
"""

import argparse
import time as _time
from dataclasses import dataclass
from datetime import time, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
from numba import njit


# ----------------------------------------------------------------------------- #
#  COSTS / PARAMS  (identical defaults to the pandas version — VERIFY all)        #
# ----------------------------------------------------------------------------- #
@dataclass
class OptCosts:
    brokerage_per_order: float = 20.0
    stt_sell_pct: float = 0.001          # options SELL, on premium (~0.10% post-Oct-2024) VERIFY
    exch_txn_pct: float = 0.0003503      # NSE options txn on premium, both sides VERIFY
    sebi_pct: float = 0.000001
    stamp_buy_pct: float = 0.00003       # buy side, on premium VERIFY
    gst_pct: float = 0.18
    slippage_points: float = 1.0         # premium pts PER LEG PER SIDE


@dataclass
class Params:
    entry_min: int = 9 * 60 + 20         # 09:20 as minutes-since-midnight
    exit_min: int = 15 * 60 + 10         # 15:10
    structure: str = "straddle"          # "straddle" | "strangle"
    strangle_offset_strikes: int = 2
    strike_step: int = 100               # Bank Nifty strike interval — VERIFY
    use_sl: bool = True
    sl_pct: float = 0.30
    lot_size: int = 15                   # !! Bank Nifty lot size has changed — VERIFY
    lots: int = 1
    capital: float = 200_000.0
    vix_max: float = float("inf")        # skip days where India VIX close > this


# ----------------------------------------------------------------------------- #
#  numba hot loop — the one genuinely sequential bit (and where trailing/combined #
#  SL logic would later live, which is exactly when numba stops being optional)   #
# ----------------------------------------------------------------------------- #
@njit(cache=True)
def _exit_leg(high, close, entry_px, use_sl, sl_pct, slip):
    """Short leg exit. Pain = premium rising -> scan bar HIGH for first SL breach.
    Returns (exit_fill_price, reason_code) reason: 0=sl, 1=time, 2=no-data."""
    n = high.shape[0]
    if n == 0:
        return entry_px + slip, 2
    if use_sl:
        thresh = entry_px * (1.0 + sl_pct)
        for i in range(n):
            if high[i] >= thresh:
                return thresh + slip, 0
    return close[n - 1] + slip, 1


# ----------------------------------------------------------------------------- #
#  DATA (Polars)                                                                 #
# ----------------------------------------------------------------------------- #
def _normalize(df: pl.DataFrame) -> pl.DataFrame:
    df = df.rename({c: c.strip().lower() for c in df.columns})

    # build datetime (parse strings; pass real datetime dtypes straight through)
    def _to_dt(colname):
        col = pl.col(colname)
        return col.str.to_datetime(strict=False) if df.schema[colname] == pl.Utf8 \
            else col.cast(pl.Datetime, strict=False)

    if "dt" in df.columns:
        df = df.with_columns(_to_dt("dt").alias("dt"))
    elif "datetime" in df.columns:
        df = df.with_columns(_to_dt("datetime").alias("dt"))
    elif "timestamp" in df.columns:
        df = df.with_columns(_to_dt("timestamp").alias("dt"))
    elif "date" in df.columns and "time" in df.columns:
        df = df.with_columns(
            (pl.col("date").cast(pl.Utf8) + " " + pl.col("time").cast(pl.Utf8))
            .str.to_datetime(strict=False).alias("dt"))
    else:
        raise ValueError("Need datetime / timestamp / date+time columns.")

    df = df.with_columns(
        pl.col("opt_type").cast(pl.Utf8).str.to_uppercase().str.strip_chars()
          .replace({"CALL": "CE", "C": "CE", "PUT": "PE", "P": "PE"}).alias("opt_type"),
        pl.col("expiry").cast(pl.Date, strict=False).alias("expiry"),
        pl.col("strike").cast(pl.Float64).round(0).cast(pl.Int64).alias("strike"),
        pl.col("close").cast(pl.Float64).alias("close"),
    )
    for c in ("open", "high", "low"):
        if c not in df.columns:
            df = df.with_columns(pl.col("close").alias(c))
        else:
            df = df.with_columns(pl.col(c).cast(pl.Float64))
    df = df.with_columns(
        pl.col("dt").dt.date().alias("date"),
        (pl.col("dt").dt.hour().cast(pl.Int32) * 60
         + pl.col("dt").dt.minute().cast(pl.Int32)).alias("mins"),
    )
    return df.select(["dt", "date", "mins", "expiry", "strike", "opt_type",
                      "open", "high", "low", "close"]).sort("dt")


def load(path: str, is_parquet: bool) -> pl.DataFrame:
    df = pl.read_parquet(path) if is_parquet else pl.read_csv(path)
    return _normalize(df)


def generate_synthetic(n_days: int = 250, seed: int = 11) -> pl.DataFrame:
    """FAKE premiums, code-test only. Mirrors the pandas version's generator."""
    rng = np.random.default_rng(seed)
    rows = []
    spot = 48_000.0
    day = datetime(2023, 1, 2, 9, 15)
    made = 0
    minutes = (15 * 60 + 15) - (9 * 60 + 15)
    while made < n_days:
        if day.weekday() < 5:
            d = day
            while d.weekday() != 3:
                d += timedelta(days=1)
            exp = d.date()
            dte = max((exp - day.date()).days, 0)
            spot *= (1 + rng.normal(0, 0.006))
            regime = rng.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
            drift = regime * rng.uniform(0.5, 3.0)
            s = spot
            t = day
            for m in range(minutes):
                s += rng.normal(drift, 25.0)
                atm = round(s / 100) * 100
                tv = 200 * np.sqrt((dte + (1 - m / minutes)) / 7.0)
                for k in range(-5, 6):
                    strike = atm + k * 100
                    for ot in ("CE", "PE"):
                        intr = max(s - strike, 0) if ot == "CE" else max(strike - s, 0)
                        prem = max(intr + tv * np.exp(-((strike - s) / 600.0) ** 2)
                                   + rng.normal(0, 3), 0.05)
                        hi = prem * (1 + abs(rng.normal(0, 0.03)))
                        lo = prem * (1 - abs(rng.normal(0, 0.03)))
                        rows.append((t, exp, int(strike), ot, prem, hi, lo, prem))
                t += timedelta(minutes=1)
            made += 1
        day += timedelta(days=1)
    df = pl.DataFrame(rows, schema=["dt", "expiry", "strike", "opt_type",
                                    "open", "high", "low", "close"], orient="row")
    df = df.with_columns(pl.col("expiry").cast(pl.Date))
    return _normalize(df)


# ----------------------------------------------------------------------------- #
#  ENGINE                                                                        #
# ----------------------------------------------------------------------------- #
def _leg_cost(premium, units, side, c: OptCosts) -> float:
    turn = premium * units
    brokerage = c.brokerage_per_order
    stt = c.stt_sell_pct * turn if side == "sell" else 0.0
    exch = c.exch_txn_pct * turn
    sebi = c.sebi_pct * turn
    stamp = c.stamp_buy_pct * turn if side == "buy" else 0.0
    gst = c.gst_pct * (brokerage + exch + sebi)
    return brokerage + stt + exch + sebi + stamp + gst


def run_backtest(df: pl.DataFrame, p: Params, c: OptCosts,
                 vix: pd.DataFrame | None = None) -> pd.DataFrame:
    units = p.lots * p.lot_size
    slip = c.slippage_points

    # Build VIX lookup {date -> vix_close} for fast per-day filtering
    vix_map: dict = {}
    if vix is not None and not vix.empty and p.vix_max < float("inf"):
        vix_map = {row.date: row.vix for row in vix.itertuples()}

    # nearest expiry >= date, vectorized, then keep only those rows
    nexp = (df.select(["date", "expiry"]).unique()
              .filter(pl.col("expiry") >= pl.col("date"))
              .group_by("date").agg(pl.col("expiry").min().alias("nexp")))
    df = df.join(nexp, on="date").filter(pl.col("expiry") == pl.col("nexp"))

    trades = []
    for (d,), day in df.partition_by("date", as_dict=True).items():
        # VIX regime filter — skip high-volatility days
        if vix_map and vix_map.get(d, 0) > p.vix_max:
            continue
        # entry snapshot
        snap = day.filter(pl.col("mins") == p.entry_min)
        if snap.height == 0:
            later = day.filter(pl.col("mins") >= p.entry_min)
            if later.height == 0:
                continue
            first = later.select(pl.col("mins").min()).item()
            snap = day.filter(pl.col("mins") == first)

        ce = snap.filter(pl.col("opt_type") == "CE").select(["strike", "close"])
        pe = snap.filter(pl.col("opt_type") == "PE").select(["strike", "close"])
        j = ce.join(pe, on="strike", suffix="_pe")
        if j.height == 0:
            continue
        j = j.with_columns((pl.col("close") - pl.col("close_pe")).abs().alias("d")) \
             .sort(["d", "strike"])
        atm = int(j.select("strike").row(0)[0])

        if p.structure == "straddle":
            ce_strike = pe_strike = atm
        else:
            ce_strike = atm + p.strangle_offset_strikes * p.strike_step
            pe_strike = atm - p.strangle_offset_strikes * p.strike_step

        def entry_prem(strike, ot):
            r = snap.filter((pl.col("strike") == strike) & (pl.col("opt_type") == ot))
            return float(r.select("close").row(0)[0]) if r.height else None

        ce0, pe0 = entry_prem(ce_strike, "CE"), entry_prem(pe_strike, "PE")
        if ce0 is None or pe0 is None or ce0 <= 0 or pe0 <= 0:
            continue

        ce_in, pe_in = ce0 - slip, pe0 - slip
        credit = ce_in + pe_in

        def leg_arrays(strike, ot):
            s = (day.filter((pl.col("strike") == strike) & (pl.col("opt_type") == ot)
                            & (pl.col("mins") > p.entry_min) & (pl.col("mins") <= p.exit_min))
                    .sort("mins"))
            return s.select("high").to_numpy().ravel(), s.select("close").to_numpy().ravel()

        ce_hi, ce_cl = leg_arrays(ce_strike, "CE")
        pe_hi, pe_cl = leg_arrays(pe_strike, "PE")

        ce_exit, ce_rc = _exit_leg(ce_hi, ce_cl, ce0, p.use_sl, p.sl_pct, slip)
        pe_exit, pe_rc = _exit_leg(pe_hi, pe_cl, pe0, p.use_sl, p.sl_pct, slip)
        reason = {0: "sl", 1: "time", 2: "noexit"}

        ce_pnl = (ce_in - ce_exit) * units
        pe_pnl = (pe_in - pe_exit) * units
        gross = ce_pnl + pe_pnl
        cost = (_leg_cost(ce0, units, "sell", c) + _leg_cost(pe0, units, "sell", c)
                + _leg_cost(ce_exit, units, "buy", c) + _leg_cost(pe_exit, units, "buy", c))
        net = gross - cost

        trades.append({
            "date": d, "atm": atm, "ce_strike": ce_strike, "pe_strike": pe_strike,
            "credit_pts": round(credit, 1),
            "ce_reason": reason[ce_rc], "pe_reason": reason[pe_rc],
            "gross_pnl": round(gross, 0), "cost": round(cost, 0), "net_pnl": round(net, 0),
        })

    return pd.DataFrame(trades)


# ----------------------------------------------------------------------------- #
#  METRICS / REPORT  (identical to the pandas version for a fair comparison)      #
# ----------------------------------------------------------------------------- #
def metrics(trades: pd.DataFrame, capital: float) -> dict:
    if trades.empty:
        return {"trades": 0}
    net = trades["net_pnl"].reset_index(drop=True)
    wins, losses = net[net > 0], net[net < 0]
    equity = capital + net.cumsum()
    dd = (equity - equity.cummax()) / equity.cummax()
    rets = (capital + net.cumsum()).pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else float("nan")
    gp, gl = wins.sum(), -losses.sum()
    worst = trades.nsmallest(5, "net_pnl")[["date", "net_pnl"]]
    worst_str = "; ".join(f"{r.date}:{int(r.net_pnl)}" for r in worst.itertuples())
    return {
        "trades": len(trades),
        "win_rate_%": round(100 * len(wins) / len(trades), 1),
        "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
        "avg_win": round(wins.mean(), 0) if len(wins) else 0,
        "avg_loss": round(losses.mean(), 0) if len(losses) else 0,
        "worst_day": round(net.min(), 0),
        "net_pnl": round(net.sum(), 0),
        "return_%": round(100 * (equity.iloc[-1] / capital - 1), 1),
        "max_dd_%": round(100 * dd.min(), 1),
        "sharpe": round(sharpe, 2),
        "worst_5_days": worst_str,
    }


def vix_sweep(df: pl.DataFrame, base: Params, c: OptCosts,
              vix: pd.DataFrame,
              thresholds: list[float] | None = None):
    """Run the best variant (Straddle +SL30) across a range of VIX thresholds
    and print a single comparison table — out-of-sample only."""
    if thresholds is None:
        thresholds = [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, float("inf")]

    dates = sorted(df.select("date").unique().to_series().to_list())
    split = dates[len(dates) // 2]
    test = df.filter(pl.col("date") >= split)

    p_straddle = Params(**{**base.__dict__, "structure": "straddle",
                           "use_sl": True, "sl_pct": 0.30})

    print(f"\n--- VIX Threshold Sweep (Straddle +SL30, out-of-sample: {split} -> {dates[-1]}) ---\n")
    rows = []
    for thresh in thresholds:
        p = Params(**{**p_straddle.__dict__, "vix_max": thresh})
        m = metrics(run_backtest(test, p, c, vix=vix), p.capital)
        label = f"VIX ≤ {thresh:.0f}" if thresh < float("inf") else "No filter"
        rows.append({"vix_max": label, **m})

    out = pd.DataFrame(rows).set_index("vix_max")
    pd.set_option("display.width", 200, "display.max_columns", 20)
    cols = ["trades", "win_rate_%", "profit_factor", "net_pnl", "return_%",
            "max_dd_%", "sharpe", "worst_day"]
    print(out[cols].to_string())
    print()

    # Highlight the best Sharpe
    valid = out["sharpe"].replace(float("nan"), -999)
    best = valid.idxmax()
    print(f"Best Sharpe: {best}  (sharpe={out.loc[best, 'sharpe']}, "
          f"net_pnl=₹{int(out.loc[best, 'net_pnl']):,})\n")


def report(df: pl.DataFrame, base: Params, c: OptCosts,
           vix: pd.DataFrame | None = None):
    dates = sorted(df.select("date").unique().to_series().to_list())
    split = dates[len(dates) // 2]
    train = df.filter(pl.col("date") < split)
    test = df.filter(pl.col("date") >= split)
    print(f"\nData: {dates[0]} -> {dates[-1]}  ({len(dates)} sessions)")
    print(f"In-sample : {dates[0]} -> {split} (excl.)   Out-sample: {split} -> {dates[-1]}")
    if vix is not None and not vix.empty and base.vix_max < float("inf"):
        print(f"VIX filter : skip days where India VIX > {base.vix_max}\n")
    else:
        print()

    variants = {
        "Straddle +SL30": Params(**{**base.__dict__, "structure": "straddle", "use_sl": True, "sl_pct": 0.30}),
        "Straddle noSL":  Params(**{**base.__dict__, "structure": "straddle", "use_sl": False}),
        "Strangle +SL30": Params(**{**base.__dict__, "structure": "strangle", "use_sl": True, "sl_pct": 0.30}),
    }
    rows = []
    t0 = _time.perf_counter()
    for name, p in variants.items():
        for label, data in (("in-sample", train), ("out-sample", test)):
            m = metrics(run_backtest(data, p, c, vix=vix), p.capital)
            rows.append({"variant": name, "period": label, **m})
    elapsed = _time.perf_counter() - t0
    out = pd.DataFrame(rows).set_index(["variant", "period"])

    pd.set_option("display.width", 240, "display.max_columns", 40)
    cols = ["trades", "win_rate_%", "profit_factor", "avg_win", "avg_loss",
            "worst_day", "net_pnl", "return_%", "max_dd_%", "sharpe"]
    print(out[cols].to_string())
    print("\n--- THE TAIL (worst 5 days, out-of-sample) ---")
    for name in variants:
        try:
            print(f"{name:16s}: {out.loc[(name, 'out-sample'), 'worst_5_days']}")
        except KeyError:
            pass
    print(f"\n[timing] 6 backtests in {elapsed:.3f}s (excludes numba warm-up & IO)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--structure", default="straddle", choices=["straddle", "strangle"])
    ap.add_argument("--lot_size", type=int, default=15)
    ap.add_argument("--lots", type=int, default=1)
    ap.add_argument("--capital", type=float, default=200_000.0)
    ap.add_argument("--vix-file", default=None,
                    help="path to VIX parquet (banknifty_chain_vix.parquet by default if --parquet given)")
    ap.add_argument("--vix-max", type=float, default=float("inf"),
                    help="skip trading days where India VIX close > this value (e.g. 18)")
    ap.add_argument("--vix-sweep", action="store_true",
                    help="sweep VIX thresholds 14-20 + no-filter and print comparison table")
    args = ap.parse_args()

    costs = OptCosts()
    params = Params(structure=args.structure, lot_size=args.lot_size,
                    lots=args.lots, capital=args.capital, vix_max=args.vix_max)

    # warm up the numba kernel so its one-time compile isn't blamed on the strategy
    _exit_leg(np.array([1.0]), np.array([1.0]), 1.0, True, 0.3, 1.0)

    t_io = _time.perf_counter()
    if args.parquet:
        df = load(args.parquet, is_parquet=True)
    elif args.csv:
        df = load(args.csv, is_parquet=False)
        cache = args.csv.rsplit(".", 1)[0] + ".parquet"
        df.write_parquet(cache)
        print(f"[io] cached to {cache} — next time use --parquet {cache} for the fast path")
    else:
        print("\n*** NO DATA — SYNTHETIC chain, FAKE premiums. Code-test only. ***")
        df = generate_synthetic()
    print(f"[io] load+normalize in {_time.perf_counter() - t_io:.3f}s")

    # Load VIX data — auto-detect alongside parquet if not explicitly given
    vix_df = None
    vix_path = args.vix_file
    if vix_path is None and args.parquet:
        vix_path = args.parquet.replace(".parquet", "_vix.parquet")
    if vix_path:
        import os
        if os.path.exists(vix_path):
            vix_df = pd.read_parquet(vix_path)
            print(f"[io] loaded VIX data: {len(vix_df)} days from {vix_path}")
        elif args.vix_max < float("inf"):
            print(f"[warn] --vix-max set but VIX file not found at {vix_path} — filter disabled")

    if args.vix_sweep:
        if vix_df is None:
            print("[error] --vix-sweep requires VIX data — run angel_fetch.py first")
        else:
            vix_sweep(df, params, costs, vix_df)
    else:
        report(df, params, costs, vix=vix_df)


if __name__ == "__main__":
    main()
