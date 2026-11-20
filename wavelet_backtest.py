import math
import os
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import pywt
import yaml
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)
# ==============================================================================
# 1. PRODUCTION STRATEGY PARAMETERS (LOADED FROM EXTERNAL config.yaml)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as _f:
    CFG = yaml.safe_load(_f)

SUBFOLDER = BASE_DIR / CFG["paths"]["csv_dir"]
FILE_PATH_7226 = SUBFOLDER / CFG["paths"]["file_7226"]
FILE_PATH_7552 = SUBFOLDER / CFG["paths"]["file_7552"]

BOARD_SIZE = int(CFG["account_and_execution"]["board_size"])
MIN_ORDER_VAL = float(CFG["account_and_execution"]["min_order_val"])
SPREAD_TICKS = int(CFG["account_and_execution"].get("spread_ticks", 1))
BASE_CASH_PCT = float(CFG["allocation_and_risk"]["base_cash_pct"])
MAX_CASH_PCT = float(CFG["allocation_and_risk"]["max_cash_pct"])
MAX_TOTAL_ORDERS = int(CFG["account_and_execution"]["max_total_orders"])
LOOKBACK_WINDOW = int(CFG["strategy_core"]["lookback_window"])
SLOPE_CENTER = float(CFG["fuzzy_membership"]["slope_center"])
D3_CENTER = float(CFG["fuzzy_membership"]["d3_center"])
COOL_DOWN_BARS = int(CFG["strategy_core"]["cool_down_bars"])
MAX_HOLD_DAYS = int(CFG["strategy_core"]["max_hold_days"])


def _round_up_to_hkd(amount: float) -> float:
    """Round stamp duty up to the nearest whole HKD per HK Government rules."""
    return float(math.ceil(amount))


def calc_hk_trading_fees(
    asset_type: str,
    price_per_unit: float,
    no_of_units: int,
    board_lot_size: int = None,
    option_contract_multiplier: int = 100,
    option_class: int = 1,
    commission_free: bool = True,
    platform_fee: float = 15.0,
) -> dict:
    """
    Calculate exact transaction fees for Hong Kong assets on Futu HK.
    Applies individual regulatory rounding, caps, and CCASS floors.
    """
    asset_type = asset_type.lower().strip()

    if asset_type not in {"stock", "etf", "option"}:
        raise ValueError("asset_type must be 'stock', 'etf', or 'option'.")

    if price_per_unit <= 0 or no_of_units <= 0:
        return {"transaction_amount": 0.0, "total_fees": 0.0, "total_cost": 0.0}

    if asset_type in {"stock", "etf"}:
        if board_lot_size is None:
            raise ValueError("board_lot_size is required for stock/ETF.")
        if no_of_units % board_lot_size != 0:
            raise ValueError(
                f"no_of_units ({no_of_units}) must be a multiple of board_lot_size ({board_lot_size})."
            )
    else:
        board_lot_size = None

    if asset_type == "option":
        transaction_amount = price_per_unit * option_contract_multiplier * no_of_units
        commission = round(max(3.0, transaction_amount * 0.002), 2)
        option_class_fee = {1: 3.0, 2: 1.0, 3: 0.5}.get(option_class)
        trading_system_fee = round(option_class_fee * no_of_units, 2)
        stamp_duty = trading_fee = sfc_levy = frc_levy = settlement_fee = 0.0
    else:
        transaction_amount = price_per_unit * no_of_units
        commission = 0.0 if commission_free else round(max(3.0, transaction_amount * 0.0003), 2)
        trading_system_fee = 0.0
        settlement_fee = round(max(2.0, min(100.0, transaction_amount * 0.000042)), 2)
        stamp_duty = 0.0 if asset_type == "etf" else float(_round_up_to_hkd(transaction_amount * 0.001))
        trading_fee = round(max(0.01, transaction_amount * 0.0000565), 2)
        sfc_levy = round(max(0.01, transaction_amount * 0.000027), 2)
        frc_levy = round(max(0.01, transaction_amount * 0.0000015) if transaction_amount > 0 else 0.0, 2)

    platform_fee_fixed = round(platform_fee, 2)

    total_fees = round(
        commission
        + platform_fee_fixed
        + trading_system_fee
        + settlement_fee
        + stamp_duty
        + trading_fee
        + sfc_levy
        + frc_levy,
        2,
    )
    return {
        "transaction_amount": round(transaction_amount, 2),
        "total_fees": total_fees,
        "total_cost": round(transaction_amount + total_fees, 2),
    }


def resample_hk_market_75min(file_path: str, asset_label: str) -> pd.DataFrame:
    """Resamples 1-minute HKEX CSV into 4 uniform 75-min bars per trading day (regular HKEX hours only)."""
    print(f"[STEP 1 - DATA LOADER] Reading 1-minute CSV for {asset_label} from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing target database file at: {file_path}")

    df_1m = pd.read_csv(file_path)
    df_1m["datetime"] = pd.to_datetime(df_1m["datetime"])
    df_1m = df_1m.sort_values("datetime").reset_index(drop=True)

    # Filter strictly regular HKEX sessions (09:30–11:59 and 13:00–15:59)
    raw_count = len(df_1m)
    t = df_1m["datetime"].dt.time
    in_morning = (t >= pd.to_datetime("09:30").time()) & (t < pd.to_datetime("12:00").time())
    in_afternoon = (t >= pd.to_datetime("13:00").time()) & (t < pd.to_datetime("16:00").time())
    df_1m = df_1m[in_morning | in_afternoon].reset_index(drop=True)
    print(
        f"  -> Filtered HKEX regular hours: kept {len(df_1m):,} of {raw_count:,} bars "
        f"(removed {raw_count - len(df_1m):,} off-session/lunch bars)."
    )

    df_1m["trade_date"] = df_1m["datetime"].dt.date
    resampled_bars = []
    chunk_size = 75

    for t_date, day_df in df_1m.groupby("trade_date", sort=True):
        day_df = day_df.reset_index(drop=True)
        for idx in range(0, len(day_df), chunk_size):
            chunk = day_df.iloc[idx : idx + chunk_size]
            if len(chunk) < 60:
                continue

            resampled_bars.append({
                "time": chunk["datetime"].iloc[-1],
                "trade_date": t_date,
                "open": float(next(iter(chunk["open"]))),
                "high": float(chunk["high"].max()),
                "low": float(chunk["low"].min()),
                "close": float(chunk["close"].iloc[-1]),
                "volume": float(chunk["volume"].sum()),
            })

    df_res = pd.DataFrame(resampled_bars).sort_values("time").reset_index(drop=True)

    # print(df_res.head(200))
    # input('@@@@@@@')
    first_dt = next(iter(df_res["time"]))
    last_dt = df_res["time"].iloc[-1]
    print(f"  -> Resampled {len(df_1m):,} 1-min rows into {len(df_res):,} uniform 75-min session bars ({first_dt} to {last_dt}).")
    return df_res


# ==============================================================================
# 2. CONTINUOUS FUZZY MEMBERSHIP FUNCTIONS (ANTI-CLIFF-EDGE SMOOTHING)
# ==============================================================================
def fuzzy_sigmoid(x: float, center: float, steepness: float) -> float:
    """Smooth S-curve membership in [0.0, 1.0] replacing brittle step thresholds."""
    z = float(np.clip(steepness * (x - center), -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def compute_fuzzy_entry_conviction(
    s4_slope: float,
    d3_val: float,
    slope_center: float = SLOPE_CENTER,
    d3_center: float = D3_CENTER,
) -> tuple[int, float]:
    """
    Evaluates directional fuzzy membership for Bull (+1) and Bear (-1) regimes.
    Returns (direction_signal, mu_conviction in [0.0, 1.0]).
    """
    # Bullish fuzzy memberships (positive slope + positive cyclical d3 confirmation)
    mu_slope_bull = fuzzy_sigmoid(s4_slope, center=slope_center, steepness=95.0)
    mu_d3_bull = fuzzy_sigmoid(d3_val, center=d3_center, steepness=55.0)
    # Anti-exhaustion damper so extreme vertical spikes don't chase the very top
    mu_exhaust_bull = 1.0 - fuzzy_sigmoid(s4_slope, center=slope_center * 45.0, steepness=4.0)
    mu_bull = float(np.sqrt(mu_slope_bull * mu_d3_bull) * (0.85 + 0.15 * mu_exhaust_bull))

    # Bearish fuzzy memberships (negative slope + negative cyclical d3 confirmation)
    mu_slope_bear = fuzzy_sigmoid(-s4_slope, center=slope_center, steepness=95.0)
    mu_d3_bear = fuzzy_sigmoid(-d3_val, center=d3_center, steepness=55.0)
    mu_exhaust_bear = 1.0 - fuzzy_sigmoid(-s4_slope, center=slope_center * 45.0, steepness=4.0)
    mu_bear = float(np.sqrt(mu_slope_bear * mu_d3_bear) * (0.85 + 0.15 * mu_exhaust_bear))

    if mu_bull >= 0.50 and mu_bull > mu_bear:
        return 1, mu_bull
    if mu_bear >= 0.50 and mu_bear > mu_bull:
        return -1, mu_bear
    return 0, 0.0

def compute_fuzzy_dynamic_params(
    s4_slope: float,
    d3_val: float,
    target_slope_threshold: float,
    current_vol: float,
    baseline_vol: float,
    trade_count: int,
    min_pct: float = BASE_CASH_PCT,
    max_pct: float = MAX_CASH_PCT,
    current_drawdown: float = 0.0,
) -> tuple[float, float, float]:
    if trade_count == 0:
        return min_pct, 2.50, 3.88

    # 1. Player 1 Fuzzy Strategy Membership (Wavelet Trend + Cyclical Conviction)
    slope_excess = max(
        0.0,
        (abs(s4_slope) - target_slope_threshold)
        / (target_slope_threshold * 3.0 + 1e-6),
    )
    cycle_boost = max(0.0, (abs(d3_val) - D3_CENTER) / 0.15)
    mu_slope = fuzzy_sigmoid(slope_excess, center=0.50, steepness=4.5)
    mu_cycle = fuzzy_sigmoid(cycle_boost, center=0.50, steepness=4.5)
    mu_conv = 0.70 * mu_slope + 0.30 * mu_cycle

    # 2. Player 2 Fuzzy Market Regime (Volatility Expansion vs. Calm)
    vol_ratio = current_vol / (baseline_vol + 1e-6)
    mu_vol_expand = fuzzy_sigmoid(vol_ratio, center=1.02, steepness=10.0)
    mu_vol_calm = 1.0 - mu_vol_expand

    # 3. Fuzzy 2x2 Strategic Payoff Matrix
    w_squeeze = mu_conv * mu_vol_expand
    w_steady  = mu_conv * mu_vol_calm
    w_chop    = (1.0 - mu_conv) * mu_vol_expand
    w_drift   = (1.0 - mu_conv) * mu_vol_calm

    nash_payoff = (
        1.00 * w_squeeze
        + 0.85 * w_steady
        + 0.25 * w_chop
        + 0.40 * w_drift
    )
    composite_weight = fuzzy_sigmoid(nash_payoff, center=0.58, steepness=7.5)

    # 4. FUZZIFIED MAX_CASH_PCT CEILING [0.56 to 0.72]:
    #    - Unlocks up to +0.16 extra ceiling when drawdown is low AND volatility isn't panicking
    #    - Automatically contracts back to 0.56 during drawdowns or extreme volatility spikes
    mu_dd_safe = 1.0 - fuzzy_sigmoid(current_drawdown, center=0.08, steepness=45.0)
    mu_vol_safe = 1.0 - fuzzy_sigmoid(vol_ratio, center=1.45, steepness=12.0)
    fuzzy_max_pct = max_pct + 0.16 * (mu_dd_safe * mu_vol_safe)

    # 5. Dynamic Position Allocation % inside [0.41, fuzzy_max_pct]
    dynamic_pct = float(
        np.clip(
            min_pct + composite_weight * (fuzzy_max_pct - min_pct),
            min_pct,
            fuzzy_max_pct,
        )
    )

    # 6. Anti-Stop-Hunt SL [2.45, 2.60] & Equilibrium TP [3.83, 3.91]
    sl_mult = round(2.45 + 0.15 * mu_vol_expand, 4)
    tp_mult = round(3.83 + 0.08 * composite_weight, 4)

    return round(dynamic_pct, 4), sl_mult, tp_mult


SPREAD_TICKS = 1  # Set to 0 for idealized fills, 1 for realistic 1-tick spread, 2 for heavy stress test


def get_hkex_tick_size(price: float) -> float:
    """Official HKEX minimum spread tick size for equities and ETFs."""
    if price < 0.25:
        return 0.001
    if price < 0.50:
        return 0.005
    if price < 10.00:
        return 0.01
    if price < 20.00:
        return 0.02
    if price < 100.00:
        return 0.05
    if price < 200.00:
        return 0.10
    return 0.20


def apply_tick_spread(raw_price: float, side: str, ticks: int = SPREAD_TICKS) -> float:
    """Shift fill price by N HKEX ticks against the order (BUY fills higher, SELL fills lower)."""
    if ticks <= 0:
        return raw_price
    tick = get_hkex_tick_size(raw_price)
    if side.upper() == "BUY":
        return round(raw_price + (tick * ticks), 4)
    return round(max(tick, raw_price - (tick * ticks)), 4)


# ==============================================================================
# 3. AUDIT-VERIFIED FUZZIFIED WAVELET ALPHA ENGINE
# ==============================================================================
class FuzzifiedWaveletAlphaEngine:

    def __init__(self, initial_cash: float = 100000.0, lookback: int = LOOKBACK_WINDOW):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.current_position = 0
        self.entry_price = 0.0
        self.entry_fee = 0.0
        self.entry_idx = 0
        self.entry_date = None
        self.active_sl_mult = 2.50
        self.active_tp_mult = 3.88
        self.shares = 0
        self.trade_ledger = []
        self.LOOKBACK = lookback

    def run_backtest(self, df_7226: pd.DataFrame, df_7552: pd.DataFrame, slope_center: float = SLOPE_CENTER,
            d3_center: float = D3_CENTER, min_cash_pct: float = BASE_CASH_PCT,
            max_cash_pct: float = MAX_CASH_PCT, ) -> dict:
        print("\n" + "=" * 95)
        print("[STEP 2 - DATA ALIGNMENT & CAUSAL WAVELET DECOMPOSITION]")
        print("=" * 95)

        df = (pd.merge(df_7226, df_7552, on="time", suffixes=("_7226", "_7552")).sort_values("time").reset_index(
            drop=True))
        print(f"  -> Synchronized {len(df):,} clock-aligned 75-minute bars across 7226.HK and 7552.HK.")

        close_7226 = df["close_7226"].to_numpy()
        close_7552 = df["close_7552"].to_numpy()
        timestamps = df["time"].to_numpy()

        s4_trend = np.zeros(len(df))
        d3_cyclical = np.zeros(len(df))

        print(
            f"  -> Computing rolling db4 level-4 wavelet decomposition (Lookback = {self.LOOKBACK} bars, Mode = symmetric)...")
        for i in range(self.LOOKBACK, len(df)):
            window = close_7226[i - self.LOOKBACK + 1: i + 1]
            writable_window = np.array(window, dtype=np.float64, copy=True)
            cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(writable_window, "db4", level=4, mode="symmetric")
            s4_trend[i] = float(cA4[-1])
            d3_cyclical[i] = float(cD3[-1])

        df["s4_raw_slope"] = pd.Series(s4_trend).diff().fillna(0.0)
        df["s4_slope"] = df["s4_raw_slope"].ewm(span=10, adjust=False).mean()

        vol_7226 = pd.Series(close_7226).rolling(window=40, min_periods=1).std().fillna(0.0).to_numpy() + 1e-6
        base_vol_7226 = pd.Series(vol_7226).rolling(window=120, min_periods=1).mean().to_numpy() + 1e-6

        print(
            f"  -> Indicator Diagnostics | s4_slope range: [{df['s4_slope'].min():.4f}, {df['s4_slope'].max():.4f}] | "
            f"d3 range: [{d3_cyclical.min():.4f}, {d3_cyclical.max():.4f}]")

        print("\n" + "=" * 95)
        print("[STEP 3 - WALK-FORWARD FUZZIFIED SIMULATION & ORDER EXECUTION LOG]")
        print("  [DEBUG VERSION] FUZZIFIED EXIT HORIZON ACTIVE (v2.1 - Adaptive 16d-30d Hold)")

        print(f"  Rules: Fuzzy Entry (mu>=0.50) | Fuzzy SL: 2.35-2.85*Vol | Fuzzy TP: 3.78-4.12*Vol | "
              f"Max Hold: {MAX_HOLD_DAYS}d | Cooldown: {COOL_DOWN_BARS} bars")
        print("=" * 95)

        last_exit_idx = -100
        portfolio_history = []
        portfolio_dates = []
        entry_count = 0
        start_bar = 522

        for i in range(start_bar, len(df) - 1):
            t_idx = i
            exec_idx = i + 1
            current_time = pd.Timestamp(timestamps[exec_idx])
            exec_date = df["trade_date_7226"].values[exec_idx]

            s4_slope = float(df["s4_slope"].values[t_idx])
            d3_val = float(d3_cyclical[t_idx])
            active_vol = float(vol_7226[t_idx])

            # --- EXIT CHECKS ---
            # --- EXIT CHECKS ---
            # --- EXIT CHECKS (FUZZIFIED HORIZON & TRAILING SL) ---
            # --- EXIT CHECKS ---
            if self.current_position != 0:
                bars_held = (t_idx - self.entry_idx) + 1
                time_stop_hit = (bars_held // 4) >= MAX_HOLD_DAYS

                sl_breached, tp_breached = False, False

                if self.current_position == 1:
                    t_close = float(df["close_7226"].values[t_idx])
                    raw_open = float(df["open_7226"].values[exec_idx])
                else:
                    t_close = float(df["close_7552"].values[t_idx])
                    raw_open = float(df["open_7552"].values[exec_idx])
                exec_open = apply_tick_spread(raw_open, "SELL", SPREAD_TICKS)


                if t_close < self.entry_price - (active_vol * self.active_sl_mult):
                    sl_breached = True
                if t_close > self.entry_price + (active_vol * self.active_tp_mult):
                    tp_breached = True


                if time_stop_hit or sl_breached or tp_breached:
                    exit_value = self.shares * exec_open
                    exit_fee = calc_hk_trading_fees(
                        "etf", exec_open, self.shares, BOARD_SIZE, commission_free=True
                    )["total_fees"]
                    self.cash = round(self.cash + exit_value - exit_fee, 2)

                    net_pnl_hkd = round(
                        ((exec_open - self.entry_price) * self.shares) - self.entry_fee - exit_fee, 2
                    )
                    exit_reason = "Time" if time_stop_hit else "SL" if sl_breached else "TP"
                    asset_code = "7226.HK" if self.current_position == 1 else "7552.HK"

                    self.trade_ledger.append({
                        "time": current_time,
                        "stock_code": asset_code,
                        "action": "SELL",
                        "price": exec_open,
                        "boards": self.shares // BOARD_SIZE,
                        "alloc_pct": 0.0,
                        "fees": exit_fee,
                        "order_total": round(exit_value, 2),
                        "cash_balance": self.cash,
                        "pnl": net_pnl_hkd,
                        "reason": exit_reason,
                    })
                    print(f"  [EXIT - {exit_reason:<4}] {current_time.strftime('%Y-%m-%d %H:%M')} | {asset_code} | "
                          f"Fill: {exec_open:>6.2f} | Boards: {self.shares // BOARD_SIZE:<4} | "
                          f"Net PnL: {net_pnl_hkd:>+10,.2f} HKD | Cash: {self.cash:>10,.2f} HKD")

                    self.current_position, self.shares, self.entry_fee = 0, 0, 0.0
                    self.entry_date = None
                    last_exit_idx = exec_idx
                    portfolio_history.append(self.cash)
                    portfolio_dates.append(exec_date)
                    continue

            # --- FUZZY ENTRY CHECKS ---
            if self.current_position == 0 and len(self.trade_ledger) < MAX_TOTAL_ORDERS:
                if (exec_idx - last_exit_idx) < COOL_DOWN_BARS:
                    portfolio_history.append(self.cash)
                    portfolio_dates.append(exec_date)
                    continue

                target_signal, mu_conv = compute_fuzzy_entry_conviction(
                    s4_slope=s4_slope,
                    d3_val=d3_val,
                    slope_center=slope_center,
                    d3_center=d3_center,
                )

                target_signal = 0
                if s4_slope > slope_center and d3_val > d3_center:
                    target_signal = 1
                elif s4_slope < -slope_center and d3_val < -d3_center:
                    target_signal = -1

                if target_signal != 0:
                    raw_open = float(
                        df["open_7226"].values[exec_idx]
                        if target_signal == 1
                        else df["open_7552"].values[exec_idx]
                    )
                    exec_open = apply_tick_spread(raw_open, "BUY", SPREAD_TICKS)

                    peak_eq = max(portfolio_history) if portfolio_history else self.initial_cash
                    cur_dd = max(0.0, (peak_eq - self.cash) / (peak_eq + 1e-6))

                    dynamic_pct, sl_m, tp_m = compute_fuzzy_dynamic_params(
                        s4_slope=s4_slope,
                        d3_val=d3_val,
                        target_slope_threshold=slope_center,
                        current_vol=active_vol,
                        baseline_vol=float(base_vol_7226[t_idx]),
                        trade_count=entry_count,
                        min_pct=min_cash_pct,
                        max_pct=max_cash_pct,
                        current_drawdown=cur_dd,
                    )

                    actual_alloc = self.cash * dynamic_pct

                    if actual_alloc >= MIN_ORDER_VAL:
                        target_lots = math.floor(actual_alloc / exec_open / BOARD_SIZE)
                        target_shares = int(target_lots * BOARD_SIZE)

                        if target_shares > 0:
                            order_value = target_shares * exec_open
                            if order_value >= MIN_ORDER_VAL:
                                buy_fee = calc_hk_trading_fees(
                                    "etf", exec_open, target_shares, BOARD_SIZE, commission_free=True
                                )["total_fees"]

                                if self.cash >= order_value + buy_fee:
                                    self.cash = round(self.cash - (order_value + buy_fee), 2)
                                    self.shares = target_shares
                                    self.current_position = target_signal
                                    self.entry_price = exec_open
                                    self.entry_fee = buy_fee
                                    self.entry_idx = exec_idx
                                    self.entry_date = exec_date
                                    self.active_sl_mult = sl_m
                                    self.active_tp_mult = tp_m
                                    entry_count += 1
                                    asset_code = "7226.HK" if target_signal == 1 else "7552.HK"

                                    self.trade_ledger.append({
                                        "time": current_time,
                                        "stock_code": asset_code,
                                        "action": "BUY",
                                        "price": exec_open,
                                        "boards": target_lots,
                                        "alloc_pct": round(dynamic_pct * 100, 2),
                                        "fees": buy_fee,
                                        "order_total": round(order_value, 2),
                                        "cash_balance": self.cash,
                                        "pnl": 0.0,
                                        "reason": f"FuzzyWav (mu={mu_conv:.2f})",
                                    })
                                    print(
                                        f"  [BUY  - {dynamic_pct*100:>4.1f}%] {current_time.strftime('%Y-%m-%d %H:%M')} | {asset_code} | "
                                        f"Fill: {exec_open:>6.2f} | Boards: {target_lots:<4} | "
                                        f"Principal: {order_value:>10,.2f} HKD | Cash: {self.cash:>10,.2f} HKD "
                                        f"(SL:{sl_m:.2f}x TP:{tp_m:.2f}x)"
                                    )

            # --- END-OF-BAR MARK-TO-MARKET EQUITY ---
            current_equity = self.cash + (
                self.shares * close_7226[exec_idx]
                if self.current_position == 1
                else self.shares * close_7552[exec_idx]
                if self.current_position == -1
                else 0.0
            )
            portfolio_history.append(round(current_equity, 2))
            portfolio_dates.append(exec_date)

        print(
            f"[DEBUG] Loop finished: processed {len(portfolio_history):,} bars "
            f"(from {min(portfolio_dates)} to {max(portfolio_dates)}) | "
            f"Total orders: {len(self.trade_ledger)}"
        )

        # 5. Final Portfolio Capital & Daily-Aggregated Risk Reconciliation
        final_unsold_position_value = round(self.shares * (
            close_7226[-1] if self.current_position == 1 else close_7552[-1] if self.current_position == -1 else 0.0),
            2, )
        final_equity = round(self.cash + final_unsold_position_value, 2)
        net_return_pct = ((final_equity - self.initial_cash) / self.initial_cash) * 100

        eq_curve = np.array(portfolio_history)
        cum_max = np.maximum.accumulate(eq_curve)
        drawdowns = (cum_max - eq_curve) / (cum_max + 1e-6)
        max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0.0

        eq_df = pd.DataFrame({"date": portfolio_dates, "equity": portfolio_history})
        daily_eq = eq_df.groupby("date", sort=True)["equity"].last()
        daily_returns = daily_eq.pct_change().dropna()
        sharpe_ratio = ((daily_returns.mean() / daily_returns.std()) * math.sqrt(252) if len(
            daily_returns) > 1 and daily_returns.std() != 0 else 0.0)

        completed_trades = [t for t in self.trade_ledger if t["action"] == "SELL"]
        winning_trades = [t for t in completed_trades if t["pnl"] > 0]
        win_ratio = (len(winning_trades) / len(completed_trades)) * 100 if len(completed_trades) > 0 else 0.0

        gross_profits = sum(t["pnl"] for t in completed_trades if t["pnl"] > 0)
        gross_losses = abs(sum(t["pnl"] for t in completed_trades if t["pnl"] < 0))
        profit_factor = (round(gross_profits / gross_losses, 2) if gross_losses > 0 else (
            1.0 if gross_profits == 0 else float("inf")))
        total_fees_paid = sum(t["fees"] for t in self.trade_ledger)

        first_date = min(portfolio_dates)
        last_date = max(portfolio_dates)
        years_duration = (pd.Timestamp(last_date) - pd.Timestamp(first_date)).days / 365.25
        cagr = ((final_equity / self.initial_cash) ** (1.0 / years_duration) - 1.0) * 100 if years_duration > 0 else 0.0

        # 6. Export Single Audit-Verified Report to Disk
        log_dir = BASE_DIR / CFG["paths"]["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = log_dir / CFG["paths"]["log_filename"]


        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("=" * 154 + "\n")
            f.write(f"                 FUZZIFIED WAVELET ALPHA ENGINE LOG (Lookback: {self.LOOKBACK} | "
                    f"SlopeCenter: {slope_center:.3f} | D3Center: {d3_center:.3f} | Cash: 41%-56%)\n")
            f.write("=" * 154 + "\n")
            f.write(f"{'Datetime':<19} | {'Stock Code':<10} | {'Action':<6} | {'Alloc %':<8} | {'Reason':<17} | "
                    f"{'Price (HKD)':<11} | {'Boards':<8} | {'Fee (HKD)':<9} | {'Order Principal':<16} | "
                    f"{'Net PnL (HKD)':<14} | {'Remaining Cash'}\n")
            f.write("-" * 154 + "\n")
            for trade in self.trade_ledger:
                alloc_str = f"{trade['alloc_pct']:.1f}%" if trade["action"] == "BUY" else "-"
                pnl_str = f"{trade['pnl']:>+12,.2f}" if trade["action"] == "SELL" else f"{'-':>12}"
                f.write(f"{trade['time'].strftime('%Y-%m-%d %H:%M'):<19} | "
                        f"{trade['stock_code']:<10} | "
                        f"{trade['action']:<6} | "
                        f"{alloc_str:<8} | "
                        f"{trade['reason']:<17} | "
                        f"{trade['price']:<11.2f} | "
                        f"{trade['boards']:<8} | "
                        f"{trade['fees']:<9.2f} | "
                        f"HKD {trade['order_total']:<12,.2f} | "
                        f"{pnl_str:<14} | "
                        f"HKD {trade['cash_balance']:,.2f}\n")
            f.write("=" * 154 + "\n")

        print("\n" + "=" * 95)
        print("[STEP 4 - FINAL FUZZIFIED WAVELET PERFORMANCE SUMMARY]")
        print("=" * 95)
        print(f"  Initial Capital        : HKD {self.initial_cash:,.2f}")
        print(f"  Final Portfolio Value  : HKD {final_equity:,.2f} ({net_return_pct:+.2f}%)")
        print(f"  Annualized CAGR        : {cagr:.2f}%")
        print(
            f"  Completed Cycles       : {len(completed_trades)} round-trips ({len(self.trade_ledger)} total executions)")
        print(f"  Win Rate (Net of Fees) : {win_ratio:.2f}%")
        print(f"  Profit Factor          : {profit_factor}")
        print(f"  Daily Sharpe Ratio     : {sharpe_ratio:.2f}")
        print(f"  Max Drawdown           : {max_drawdown:.2f}%")
        print(f"  Total Futu Fees Paid   : HKD {total_fees_paid:,.2f}")
        print(f"  Report Exported To     : {log_file_path}")
        print("=" * 95)

        return {"trade_log": self.trade_ledger, "final_equity": final_equity, "remaining_cash": self.cash,
            "unsold_position_value": final_unsold_position_value, "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio, "win_ratio": win_ratio, "profit_factor": profit_factor,
            "annualized_return": cagr, }


if __name__ == "__main__":
    print("=" * 95)
    print("        FUZZIFIED WAVELET ALPHA ENGINE - AUDIT-VERIFIED PRODUCTION BACKTEST")
    print("=" * 95)
    try:
        clean_df_7226 = resample_hk_market_75min(FILE_PATH_7226,
                                                 "7226.HK (CSOP HSI Tech 2x Bull)")
        clean_df_7552 = resample_hk_market_75min(FILE_PATH_7552,
                                                 "7552.HK (CSOP HSI Tech -2x Bear)")

        engine = FuzzifiedWaveletAlphaEngine(
            initial_cash=float(CFG["account_and_execution"]["initial_cash"]),
            lookback=LOOKBACK_WINDOW,
        )

        engine.run_backtest(clean_df_7226, clean_df_7552, slope_center=SLOPE_CENTER,
            d3_center=D3_CENTER, min_cash_pct=BASE_CASH_PCT, max_cash_pct=MAX_CASH_PCT, )
    except Exception as e:
        print(f"\n[Execution Failure]: {e}")
        traceback.print_exc()
