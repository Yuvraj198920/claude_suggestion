"""
Selective ORB Backtest Harness  —  "Trade-Less" Opening Range Breakout for Nifty Futures
========================================================================================

What this does
--------------
Backtests the strategy we discussed and pits it head-to-head against the plain ("raw")
Opening Range Breakout, on the SAME data, with a realistic cost model and a strict
in-sample / out-of-sample split. The whole point is to find out whether the extra
filters (volatility gate + VWAP gate) actually beat the simple version AFTER costs.
If they don't, throw them out.

Two strategies compared
-----------------------
  RAW ORB        : trade the first break of the 9:15-9:30 range, either side. No filters.
  SELECTIVE ORB  : same, but only on wide-range days (vol gate) and only when price
                   agrees with VWAP (vwap gate). Optional short-only / long-only mode.

Honest warnings
---------------
1. With no --csv, this runs on SYNTHETIC random data so you can verify the code works.
   Those numbers are MEANINGLESS for trading. Plug in real Nifty futures intraday data.
2. The cost constants below are approximate and the Nifty lot size has changed several
   times. VERIFY every number in the Costs/Params blocks against current values.
3. Intrabar exits assume the worst case (stop checked before target). That is deliberate
   pessimism — better than fooling yourself.

Data format expected (CSV)
--------------------------
A datetime column (named datetime/timestamp, or separate date + time) plus
open, high, low, close, volume. Any bar size; it is resampled to `bar_minutes`.

Run
---
  python orb_backtest.py                      # synthetic smoke test
  python orb_backtest.py --csv nifty_fut.csv  # your real data
  python orb_backtest.py --csv nifty_fut.csv --direction short   # short-only selective
"""

import argparse
from dataclasses import dataclass
from datetime import time, datetime, timedelta

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- #
#  CONFIG — verify every number here against your broker / current regulations    #
# ----------------------------------------------------------------------------- #
@dataclass
class Costs:
    brokerage_per_order: float = 20.0     # flat Rs per executed order (entry & exit each)
    stt_sell_pct: float = 0.0002          # STT, futures, SELL side only (~0.02% post-Oct-2024) VERIFY
    exch_txn_pct: float = 0.0000173       # NSE futures txn charge, both sides VERIFY
    sebi_pct: float = 0.000001            # SEBI charges (Rs 10 / crore), both sides
    stamp_buy_pct: float = 0.00002        # stamp duty, BUY side only (0.002%) VERIFY
    gst_pct: float = 0.18                 # GST on (brokerage + exch + sebi)
    slippage_points: float = 1.0          # index points lost to slippage, PER SIDE


@dataclass
class Params:
    # session structure
    or_start: time = time(9, 15)
    or_end: time = time(9, 30)            # opening range = first 15 minutes
    hard_exit: time = time(14, 30)        # edge decays into the afternoon -> flat by 14:30
    bar_minutes: int = 5

    # trade management
    rr_target: float = 2.0                # target = 2R
    move_to_be_at_R: float = 1.0          # move stop to breakeven once +1R is reached
    risk_pct: float = 0.01               # risk 1% of capital per trade

    # filters
    use_vol_gate: bool = True
    vol_lookback: int = 20                # trailing sessions for the OR-width distribution
    vol_pct_threshold: float = 0.66       # only trade if today's OR width >= 66th pctile
    use_vwap_gate: bool = True
    direction_mode: str = "both"          # "both" | "long" | "short"

    # contract / capital
    lot_size: int = 75                    # !! Nifty lot size has changed repeatedly — VERIFY
    point_value: float = 1.0              # Rs per index point per unit
    starting_capital: float = 500_000.0


# ----------------------------------------------------------------------------- #
#  DATA                                                                          #
# ----------------------------------------------------------------------------- #
def load_csv(path: str, bar_minutes: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # find / build a datetime column
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
    elif "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"])
    elif "date" in df.columns and "time" in df.columns:
        dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"])
    else:
        raise ValueError("No datetime/timestamp or date+time columns found.")

    out = pd.DataFrame({
        "dt": dt,
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float) if "volume" in df.columns else 0.0,
    }).sort_values("dt").reset_index(drop=True)

    return _resample(out, bar_minutes)


def _resample(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """Resample to `bar_minutes` only if the data is finer than that."""
    deltas = df["dt"].diff().dropna()
    if deltas.empty:
        return df
    median_min = deltas.median().total_seconds() / 60.0
    if median_min >= bar_minutes:           # already coarse enough
        return df
    g = (df.set_index("dt")
           .resample(f"{bar_minutes}min", label="left", closed="left")
           .agg({"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"})
           .dropna(subset=["open"]))
    return g.reset_index()


def generate_synthetic(n_days: int = 400, seed: int = 7,
                       bar_minutes: int = 5) -> pd.DataFrame:
    """Random-walk intraday bars with per-day regimes. FOR CODE TESTING ONLY."""
    rng = np.random.default_rng(seed)
    rows = []
    price = 22_000.0
    day = datetime(2022, 1, 3, 9, 15)
    bars_per_day = int(((15 * 60 + 15) - (9 * 60 + 15)) / bar_minutes)  # 9:15 -> 15:15
    made = 0
    while made < n_days:
        if day.weekday() < 5:               # weekdays only
            regime = rng.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            drift = regime * rng.uniform(0.2, 1.4)
            t = day
            price *= (1 + rng.normal(0, 0.004))   # overnight gap
            for _ in range(bars_per_day):
                step = rng.normal(drift, 12.0)
                o = price
                c = price + step
                hi = max(o, c) + abs(rng.normal(0, 5))
                lo = min(o, c) - abs(rng.normal(0, 5))
                vol = abs(rng.normal(50_000, 15_000))
                rows.append((t, o, hi, lo, c, vol))
                price = c
                t += timedelta(minutes=bar_minutes)
            made += 1
        day += timedelta(days=1)
    return pd.DataFrame(rows, columns=["dt", "open", "high", "low", "close", "volume"])


# ----------------------------------------------------------------------------- #
#  ENGINE                                                                        #
# ----------------------------------------------------------------------------- #
def _vwap(day_df: pd.DataFrame) -> pd.Series:
    tp = (day_df["high"] + day_df["low"] + day_df["close"]) / 3.0
    cum_vol = day_df["volume"].cumsum()
    cum_tpv = (tp * day_df["volume"]).cumsum()
    vwap = cum_tpv / cum_vol.replace(0, np.nan)
    # fallback when volume is missing/zero: running mean of close
    return vwap.fillna(day_df["close"].expanding().mean())


def _trade_cost(entry, exit_, units, direction, costs: Costs) -> float:
    buy_turn = entry * units if direction == "long" else exit_ * units
    sell_turn = exit_ * units if direction == "long" else entry * units
    brokerage = 2 * costs.brokerage_per_order
    stt = costs.stt_sell_pct * sell_turn
    exch = costs.exch_txn_pct * (buy_turn + sell_turn)
    sebi = costs.sebi_pct * (buy_turn + sell_turn)
    stamp = costs.stamp_buy_pct * buy_turn
    gst = costs.gst_pct * (brokerage + exch + sebi)
    slip = costs.slippage_points * units * 2  # both sides, point_value folded below
    return brokerage + stt + exch + sebi + stamp + gst + slip


def run_backtest(df: pd.DataFrame, p: Params, costs: Costs) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["dt"].dt.date
    df["tm"] = df["dt"].dt.time

    # ---- precompute OR width per day for the volatility gate ----
    or_width = {}
    for d, g in df.groupby("date"):
        ob = g[(g["tm"] >= p.or_start) & (g["tm"] < p.or_end)]
        if len(ob):
            or_width[d] = ob["high"].max() - ob["low"].min()
    days = sorted(or_width.keys())
    width_hist = pd.Series({d: or_width[d] for d in days})

    capital = p.starting_capital
    trades = []

    for i, d in enumerate(days):
        g = df[df["date"] == d].reset_index(drop=True)
        ob = g[(g["tm"] >= p.or_start) & (g["tm"] < p.or_end)]
        if len(ob) == 0:
            continue
        or_high, or_low = ob["high"].max(), ob["low"].min()
        width = or_high - or_low
        if width <= 0:
            continue

        # volatility gate: today's width vs trailing distribution (excludes today)
        if p.use_vol_gate:
            prior = width_hist.iloc[max(0, i - p.vol_lookback):i]
            if len(prior) < max(10, p.vol_lookback // 2):
                continue                                  # warm-up
            if width < np.quantile(prior.values, p.vol_pct_threshold):
                continue

        g = g.copy()
        g["vwap"] = _vwap(g)
        post = g[(g["tm"] >= p.or_end) & (g["tm"] < p.hard_exit)].reset_index(drop=True)
        if len(post) == 0:
            continue

        # ---- find first valid breakout ----
        entry = stop = target = None
        direction = None
        entry_idx = None
        for j in range(len(post)):
            row = post.iloc[j]
            long_ok = (row["close"] > or_high and
                       (not p.use_vwap_gate or row["close"] > row["vwap"]) and
                       p.direction_mode in ("both", "long"))
            short_ok = (row["close"] < or_low and
                        (not p.use_vwap_gate or row["close"] < row["vwap"]) and
                        p.direction_mode in ("both", "short"))
            if long_ok:
                direction, entry, stop = "long", row["close"], or_low
                target = entry + p.rr_target * (entry - stop)
                entry_idx = j
                break
            if short_ok:
                direction, entry, stop = "short", row["close"], or_high
                target = entry - p.rr_target * (stop - entry)
                entry_idx = j
                break
        if entry is None:
            continue

        risk_pts = abs(entry - stop)
        if risk_pts <= 0:
            continue

        # ---- position sizing: fixed-fractional, rounded to whole lots ----
        rupee_risk = capital * p.risk_pct
        lots = int(rupee_risk // (risk_pts * p.lot_size * p.point_value))
        if lots < 1:
            continue                                       # stop too wide for risk budget
        units = lots * p.lot_size

        # ---- walk forward bar by bar from the bar AFTER entry ----
        be_moved = False
        exit_price, exit_reason = None, None
        for k in range(entry_idx + 1, len(post)):
            bar = post.iloc[k]
            if direction == "long":
                if bar["low"] <= stop:                     # pessimistic: stop first
                    exit_price, exit_reason = stop, "stop"; break
                if bar["high"] >= target:
                    exit_price, exit_reason = target, "target"; break
                if bar["tm"] >= p.hard_exit:
                    exit_price, exit_reason = bar["close"], "time"; break
                if not be_moved and bar["high"] >= entry + p.move_to_be_at_R * risk_pts:
                    stop, be_moved = entry, True
            else:
                if bar["high"] >= stop:
                    exit_price, exit_reason = stop, "stop"; break
                if bar["low"] <= target:
                    exit_price, exit_reason = target, "target"; break
                if bar["tm"] >= p.hard_exit:
                    exit_price, exit_reason = bar["close"], "time"; break
                if not be_moved and bar["low"] <= entry - p.move_to_be_at_R * risk_pts:
                    stop, be_moved = entry, True
        if exit_price is None:                             # ran out of bars
            exit_price, exit_reason = post.iloc[-1]["close"], "eod"

        gross_pts = (exit_price - entry) if direction == "long" else (entry - exit_price)
        gross_pnl = gross_pts * units * p.point_value
        cost = _trade_cost(entry, exit_price, units, direction, costs)
        net_pnl = gross_pnl - cost
        capital += net_pnl

        trades.append({
            "date": d, "direction": direction, "lots": lots, "units": units,
            "entry": round(entry, 2), "stop_init": round(or_low if direction == "long" else or_high, 2),
            "target": round(target, 2), "exit": round(exit_price, 2), "exit_reason": exit_reason,
            "risk_pts": round(risk_pts, 2), "gross_pts": round(gross_pts, 2),
            "gross_pnl": round(gross_pnl, 1), "cost": round(cost, 1),
            "net_pnl": round(net_pnl, 1),
            "R_gross": round(gross_pts / risk_pts, 3),
            "R_net": round(net_pnl / rupee_risk, 3),
            "capital": round(capital, 1),
        })

    return pd.DataFrame(trades)


# ----------------------------------------------------------------------------- #
#  METRICS                                                                       #
# ----------------------------------------------------------------------------- #
def metrics(trades: pd.DataFrame, starting_capital: float) -> dict:
    if trades.empty:
        return {"trades": 0}
    net = trades["net_pnl"]
    wins, losses = net[net > 0], net[net < 0]
    equity = starting_capital + net.cumsum()
    peak = equity.cummax()
    dd = (equity - peak) / peak

    # Sharpe from per-day P&L
    daily = trades.groupby("date")["net_pnl"].sum()
    eq_daily = starting_capital + daily.cumsum()
    rets = eq_daily.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else float("nan")

    gp, gl = wins.sum(), -losses.sum()
    return {
        "trades": len(trades),
        "win_rate_%": round(100 * len(wins) / len(trades), 1),
        "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
        "expectancy_R": round(trades["R_net"].mean(), 3),
        "avg_win": round(wins.mean(), 0) if len(wins) else 0,
        "avg_loss": round(losses.mean(), 0) if len(losses) else 0,
        "net_pnl": round(net.sum(), 0),
        "return_%": round(100 * (equity.iloc[-1] / starting_capital - 1), 1),
        "max_dd_%": round(100 * dd.min(), 1),
        "sharpe": round(sharpe, 2),
    }


# ----------------------------------------------------------------------------- #
#  REPORT                                                                        #
# ----------------------------------------------------------------------------- #
def report(df: pd.DataFrame, base: Params, costs: Costs):
    dates = sorted(df["dt"].dt.date.unique())
    split = dates[len(dates) // 2]
    train = df[df["dt"].dt.date < split]
    test = df[df["dt"].dt.date >= split]
    print(f"\nData: {dates[0]} -> {dates[-1]}  ({len(dates)} sessions)")
    print(f"In-sample  : {dates[0]} -> {split} (excl.)")
    print(f"Out-sample : {split} -> {dates[-1]}\n")

    raw = Params(**{**base.__dict__, "use_vol_gate": False, "use_vwap_gate": False,
                    "direction_mode": "both"})
    configs = {"RAW ORB": raw, "SELECTIVE ORB": base}

    rows = []
    for name, p in configs.items():
        for label, data in (("in-sample", train), ("out-sample", test)):
            m = metrics(run_backtest(data, p, costs), p.starting_capital)
            m = {"strategy": name, "period": label, **m}
            rows.append(m)
    out = pd.DataFrame(rows).set_index(["strategy", "period"])
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(out.to_string())
    print("\nRead it like this: does SELECTIVE beat RAW *out-of-sample* after costs?")
    print("If the edge only shows in-sample, the filters are curve-fit. Be ruthless.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="intraday CSV; omit for synthetic test")
    ap.add_argument("--direction", default="both", choices=["both", "long", "short"])
    ap.add_argument("--lot_size", type=int, default=75)
    ap.add_argument("--capital", type=float, default=500_000.0)
    args = ap.parse_args()

    costs = Costs()
    params = Params(direction_mode=args.direction, lot_size=args.lot_size,
                    starting_capital=args.capital)

    if args.csv:
        df = load_csv(args.csv, params.bar_minutes)
    else:
        print("\n*** NO CSV GIVEN — running on SYNTHETIC random data. "
              "Results are meaningless; use only to confirm the code runs. ***")
        df = generate_synthetic(bar_minutes=params.bar_minutes)

    report(df, params, costs)


if __name__ == "__main__":
    main()
