"""
Intraday Premium-Selling Backtest  —  Short Straddle / Strangle on Bank Nifty
=============================================================================

Built for REAL per-minute option-chain data. No Black-Scholes guessing of premiums.

What it tests
-------------
Sell ATM CE + ATM PE (straddle) or OTM CE + OTM PE (strangle) at a fixed time,
manage each leg with a per-leg stop-loss, square off by a fixed exit time.
This is the 9:20-straddle family. It wins often and loses rarely-but-large, so the
report is built around the TAIL: worst days, loss distribution, and whether the
edge survives costs out-of-sample. A high win rate here is not safety.

Honest gating fact
------------------
Premium selling has negative skew. Win rate WILL look good. Judge it by:
  - profit factor and net P&L AFTER costs, out-of-sample
  - max drawdown and the worst handful of days (printed explicitly)
  - whether a stop-loss actually helps or just bleeds you via whipsaw
If the worst 5 days wipe out months of gains, that is the real risk, not the win %.

Data schema expected (after the loader maps your columns)
--------------------------------------------------------
One row per option per minute, columns:
  dt (datetime, minute)  expiry (date)  strike (number)  opt_type ('CE'/'PE')
  open high low close (premium)  volume (optional)
Many vendors pack this into one 'symbol' string (e.g. BANKNIFTY24D19...CE).
If so, parse it into the columns above in load_csv() — a hook is marked there.
The engine finds ATM by put-call parity (strike where |CE-PE| is smallest),
so it does NOT need a separate spot feed.

Run
---
  python options_premium_backtest.py                 # synthetic smoke test
  python options_premium_backtest.py --csv chain.csv # your real Bank Nifty chain
"""

import argparse
from dataclasses import dataclass
from datetime import time, datetime, timedelta

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- #
#  COSTS — options rates differ from futures. VERIFY every number.                #
# ----------------------------------------------------------------------------- #
@dataclass
class OptCosts:
    brokerage_per_order: float = 20.0     # flat Rs per executed order
    stt_sell_pct: float = 0.001           # STT on options SELL, on premium (~0.10% post-Oct-2024) VERIFY
    exch_txn_pct: float = 0.0003503       # NSE options txn charge on premium, both sides VERIFY
    sebi_pct: float = 0.000001            # SEBI charges on premium turnover
    stamp_buy_pct: float = 0.00003        # stamp duty, BUY side, on premium (0.003%) VERIFY
    gst_pct: float = 0.18                 # GST on (brokerage + exch + sebi)
    slippage_points: float = 1.0          # premium points lost PER LEG PER SIDE (Bank Nifty spreads are wide!)


@dataclass
class Params:
    entry_time: time = time(9, 20)
    exit_time: time = time(15, 10)

    structure: str = "straddle"           # "straddle" (ATM) or "strangle"
    strangle_offset_strikes: int = 2      # for strangle: sell CE at ATM+n, PE at ATM-n
    strike_step: int = 100                # Bank Nifty strike interval — VERIFY (often 100)

    use_sl: bool = True
    sl_pct: float = 0.30                  # per-leg stop: buy back a leg if its premium rises 30%

    lot_size: int = 15                    # !! Bank Nifty lot size has changed repeatedly — VERIFY
    lots: int = 1
    capital: float = 200_000.0            # for return %; set to your deployed margin
    one_expiry_only: str = "nearest"      # use nearest weekly expiry >= trade date


# ----------------------------------------------------------------------------- #
#  DATA                                                                          #
# ----------------------------------------------------------------------------- #
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # ---- HOOK: if your data encodes everything in one 'symbol' column, parse it here ----
    # Example: split 'BANKNIFTY<expiry><strike><CE/PE>' into expiry/strike/opt_type.
    # Left as separate-columns by default.

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
    elif "date" in df.columns and "time" in df.columns:
        dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    elif "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("Need datetime / timestamp / date+time columns.")

    opt = df["opt_type"].astype(str).str.upper().str.strip()
    opt = opt.replace({"CALL": "CE", "C": "CE", "PUT": "PE", "P": "PE"})

    out = pd.DataFrame({
        "dt": dt,
        "expiry": pd.to_datetime(df["expiry"]).dt.date,
        "strike": df["strike"].astype(float).round().astype(int),
        "opt_type": opt,
        "open": df.get("open", df["close"]).astype(float),
        "high": df.get("high", df["close"]).astype(float),
        "low": df.get("low", df["close"]).astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float) if "volume" in df.columns else 0.0,
    })
    out["date"] = out["dt"].dt.date
    out["tm"] = out["dt"].dt.time
    return out.sort_values("dt").reset_index(drop=True)


def generate_synthetic(n_days: int = 250, seed: int = 11) -> pd.DataFrame:
    """Crude per-minute chain. FOR CODE TESTING ONLY — premiums are not realistic."""
    rng = np.random.default_rng(seed)
    rows = []
    spot = 48_000.0
    day = datetime(2023, 1, 2, 9, 15)
    made = 0
    minutes = int(((15 * 60 + 15) - (9 * 60 + 15)))  # 9:15 -> 15:15
    while made < n_days:
        if day.weekday() < 5:
            # weekly expiry = the coming Thursday (weekday 3)
            exp = day.date()
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
                frac_day_left = 1 - m / minutes
                tv_base = 200 * np.sqrt((dte + frac_day_left) / 7.0)  # crude time value
                for k in range(-5, 6):
                    strike = atm + k * 100
                    for ot in ("CE", "PE"):
                        intr = max(s - strike, 0) if ot == "CE" else max(strike - s, 0)
                        bump = np.exp(-((strike - s) / 600.0) ** 2)
                        prem = max(intr + tv_base * bump + rng.normal(0, 3), 0.05)
                        hi = prem * (1 + abs(rng.normal(0, 0.03)))
                        lo = prem * (1 - abs(rng.normal(0, 0.03)))
                        rows.append((t, exp, strike, ot, prem, hi, lo, prem,
                                     abs(rng.normal(1000, 300))))
                t += timedelta(minutes=1)
            made += 1
        day += timedelta(days=1)
    df = pd.DataFrame(rows, columns=["dt", "expiry", "strike", "opt_type",
                                     "open", "high", "low", "close", "volume"])
    df["date"] = df["dt"].dt.date
    df["tm"] = df["dt"].dt.time
    return df


# ----------------------------------------------------------------------------- #
#  ENGINE                                                                        #
# ----------------------------------------------------------------------------- #
def _pick_expiry(day_df: pd.DataFrame, trade_date) -> object:
    exps = sorted(e for e in day_df["expiry"].unique() if e >= trade_date)
    return exps[0] if exps else None


def _atm_strike(snap: pd.DataFrame) -> int:
    """ATM = strike minimizing |CE - PE| (put-call parity), needs no spot feed."""
    ce = snap[snap["opt_type"] == "CE"].set_index("strike")["close"]
    pe = snap[snap["opt_type"] == "PE"].set_index("strike")["close"]
    common = ce.index.intersection(pe.index)
    if len(common) == 0:
        return None
    diff = (ce[common] - pe[common]).abs()
    return int(diff.idxmin())


def _leg_cost(premium, units, side, c: OptCosts) -> float:
    turn = premium * units
    brokerage = c.brokerage_per_order
    stt = c.stt_sell_pct * turn if side == "sell" else 0.0
    exch = c.exch_txn_pct * turn
    sebi = c.sebi_pct * turn
    stamp = c.stamp_buy_pct * turn if side == "buy" else 0.0
    gst = c.gst_pct * (brokerage + exch + sebi)
    return brokerage + stt + exch + sebi + stamp + gst


def run_backtest(df: pd.DataFrame, p: Params, c: OptCosts) -> pd.DataFrame:
    units = p.lots * p.lot_size
    slip = c.slippage_points
    trades = []

    for d, g in df.groupby("date"):
        exp = _pick_expiry(g, d)
        if exp is None:
            continue
        ge = g[g["expiry"] == exp]

        entry_snap = ge[ge["tm"] == p.entry_time]
        if entry_snap.empty:
            entry_snap = ge[ge["tm"] >= p.entry_time]
            if entry_snap.empty:
                continue
            first_t = entry_snap["tm"].min()
            entry_snap = ge[ge["tm"] == first_t]

        atm = _atm_strike(entry_snap)
        if atm is None:
            continue

        if p.structure == "straddle":
            ce_strike = pe_strike = atm
        else:
            ce_strike = atm + p.strangle_offset_strikes * p.strike_step
            pe_strike = atm - p.strangle_offset_strikes * p.strike_step

        def entry_prem(strike, ot):
            r = entry_snap[(entry_snap["strike"] == strike) & (entry_snap["opt_type"] == ot)]
            return float(r["close"].iloc[0]) if len(r) else None

        ce0, pe0 = entry_prem(ce_strike, "CE"), entry_prem(pe_strike, "PE")
        if ce0 is None or pe0 is None or ce0 <= 0 or pe0 <= 0:
            continue

        # SELL both legs (slippage: we receive a bit less)
        ce_fill_in, pe_fill_in = ce0 - slip, pe0 - slip
        credit = ce_fill_in + pe_fill_in

        # minute series for each leg, from after entry to exit_time
        def leg_series(strike, ot):
            s = ge[(ge["strike"] == strike) & (ge["opt_type"] == ot) &
                   (ge["tm"] > p.entry_time) & (ge["tm"] <= p.exit_time)]
            return s.sort_values("dt")

        ce_s, pe_s = leg_series(ce_strike, "CE"), leg_series(pe_strike, "PE")

        def close_leg(series, entry_px):
            """Return (exit_fill_price, reason). For a SHORT leg, pain = premium rising;
            use bar HIGH for pessimistic SL detection. Buy-back fill adds slippage."""
            if p.use_sl:
                thresh = entry_px * (1 + p.sl_pct)
                hit = series[series["high"] >= thresh]
                if len(hit):
                    return thresh + slip, "sl"
            if len(series):
                return float(series["close"].iloc[-1]) + slip, "time"
            return entry_px + slip, "noexit"   # no data after entry -> flat-ish

        ce_exit, ce_reason = close_leg(ce_s, ce0)
        pe_exit, pe_reason = close_leg(pe_s, pe0)

        # P&L per short leg = sell_fill - buy_fill
        ce_pnl = (ce_fill_in - ce_exit) * units
        pe_pnl = (pe_fill_in - pe_exit) * units
        gross = ce_pnl + pe_pnl

        cost = (_leg_cost(ce0, units, "sell", c) + _leg_cost(pe0, units, "sell", c)
                + _leg_cost(ce_exit, units, "buy", c) + _leg_cost(pe_exit, units, "buy", c))
        net = gross - cost

        trades.append({
            "date": d, "expiry": exp, "atm": atm,
            "ce_strike": ce_strike, "pe_strike": pe_strike,
            "credit_pts": round(credit, 1),
            "ce_reason": ce_reason, "pe_reason": pe_reason,
            "gross_pnl": round(gross, 0), "cost": round(cost, 0),
            "net_pnl": round(net, 0),
        })

    return pd.DataFrame(trades)


# ----------------------------------------------------------------------------- #
#  METRICS  (tail-focused)                                                       #
# ----------------------------------------------------------------------------- #
def metrics(trades: pd.DataFrame, capital: float) -> dict:
    if trades.empty:
        return {"trades": 0}
    net = trades["net_pnl"].reset_index(drop=True)
    wins, losses = net[net > 0], net[net < 0]
    equity = capital + net.cumsum()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    daily = net  # one trade/day
    rets = (capital + daily.cumsum()).pct_change().dropna()
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


# ----------------------------------------------------------------------------- #
#  REPORT                                                                        #
# ----------------------------------------------------------------------------- #
def report(df: pd.DataFrame, base: Params, c: OptCosts):
    dates = sorted(df["date"].unique())
    split = dates[len(dates) // 2]
    train = df[df["date"] < split]
    test = df[df["date"] >= split]
    print(f"\nData: {dates[0]} -> {dates[-1]}  ({len(dates)} sessions)")
    print(f"In-sample : {dates[0]} -> {split} (excl.)   Out-sample: {split} -> {dates[-1]}\n")

    variants = {
        "Straddle +SL30":  Params(**{**base.__dict__, "structure": "straddle", "use_sl": True, "sl_pct": 0.30}),
        "Straddle noSL":   Params(**{**base.__dict__, "structure": "straddle", "use_sl": False}),
        "Strangle +SL30":  Params(**{**base.__dict__, "structure": "strangle", "use_sl": True, "sl_pct": 0.30}),
    }
    rows = []
    for name, p in variants.items():
        for label, data in (("in-sample", train), ("out-sample", test)):
            m = metrics(run_backtest(data, p, c), p.capital)
            rows.append({"variant": name, "period": label, **m})
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
    print("\nHow to read it: the win rate will look great. Ignore it. Ask instead -")
    print("  * does 'noSL' beat '+SL30'? (then your stop is just whipsaw-bleed)")
    print("  * how many average-winning days does ONE worst_day erase?")
    print("  * does any variant survive out-of-sample after costs?\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--structure", default="straddle", choices=["straddle", "strangle"])
    ap.add_argument("--lot_size", type=int, default=15)
    ap.add_argument("--lots", type=int, default=1)
    ap.add_argument("--capital", type=float, default=200_000.0)
    args = ap.parse_args()

    costs = OptCosts()
    params = Params(structure=args.structure, lot_size=args.lot_size,
                    lots=args.lots, capital=args.capital)

    if args.csv:
        df = load_csv(args.csv)
    else:
        print("\n*** NO CSV — SYNTHETIC chain with FAKE premiums. "
              "Numbers are meaningless; only proves the code runs. ***")
        df = generate_synthetic()

    report(df, params, costs)


if __name__ == "__main__":
    main()
