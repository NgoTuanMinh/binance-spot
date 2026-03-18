#!/usr/bin/env python3
"""
Backtest cho chiến lược signal bot (bộ lọc 3 lớp).

Đọc dữ liệu CSV từ binance-data-downloader:
  {DATA_DIR}/{SYMBOL}_1h.csv  ← dữ liệu chính (1H OHLCV)
  {DATA_DIR}/{SYMBOL}_1d.csv  ← nếu có; nếu không thì tự resample từ 1H

Cấu trúc CSV đầu vào (6 cột):
  timestamp, open, high, low, close, volume

Cách chạy:
  python backtest.py
  python backtest.py --data-dir ../binance-data-downloader/data/processed
  python backtest.py --symbols BTCUSDT ETHUSDT SOLUSDT --start 2022-01-01
  python backtest.py --no-btc-filter --timeout-days 20 --output-dir ./results
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────
# Tham số chiến lược (giữ đồng bộ với config.py và signal_detector.py)
# ──────────────────────────────────────────────────────────
EMA_PERIODS                 = (20, 50, 200)
RSI_PERIOD                  = 14
ATR_PERIOD                  = 14
VOLUME_SMA_PERIOD           = 20
VOLUME_SPIKE_MIN_RATIO      = 2.0
VOLUME_SPIKE_LOOKBACK       = 2       # số nến 1H đã đóng để kiểm tra spike
ACCUMULATION_LOOKBACK       = 30      # nến daily
ACCUMULATION_RANGE_MAX_PCT  = 50.0
ACCUMULATION_POSITION_PCT   = 30.0
BREAKOUT_THRESHOLD_PCT      = 95.0
ATR_SL_MULTIPLIER           = 2.0
RSI_MIN_DAILY               = 40
TP1_RR_RATIO                = 1.5
TP2_RR_RATIO                = 3.0
BTC_SYMBOL                  = "BTCUSDT"

# Tham số backtest
TRADE_TIMEOUT_DAYS          = 30      # đóng lệnh sau N ngày nếu không chạm SL/TP
RISK_PER_TRADE_PCT          = 1.0     # % equity rủi ro mỗi lệnh (để tính equity curve)
WARMUP_DAILY_BARS           = 250     # số nến daily tối thiểu trước khi bắt đầu tín hiệu
MAX_RISK_PER_TRADE_PCT      = 10.0     # maximum risk per trade in percentage

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Indicator helpers (giữ nguyên logic từ signal_detector.py)
# ──────────────────────────────────────────────────────────

def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def _rsi(s: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(window=period).mean()


# ──────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────

def load_ohlcv(data_dir: Path, symbol: str, interval: str) -> pd.DataFrame | None:
    """Load CSV từ binance-data-downloader. Trả về None nếu file không tồn tại hoặc lỗi."""
    path = data_dir / f"{symbol}_{interval}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])

        # Kiểm tra timestamp có parse thành datetime không
        if "timestamp" not in df.columns:
            logger.error("Missing 'timestamp' column in %s", path)
            return None
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            # Thử parse lại thủ công (đề phòng format khác như Unix ms)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)

        # Loại bỏ row có timestamp không parse được (NaT) trước khi set_index
        nat_count = df["timestamp"].isna().sum()
        if nat_count > 0:
            logger.warning("%s: dropped %d rows with invalid timestamp", path.name, nat_count)
            df = df[df["timestamp"].notna()]

        if df.empty:
            logger.error("%s: no valid rows after timestamp filter", path.name)
            return None

        df.set_index("timestamp", inplace=True)

        # Đảm bảo index là DatetimeIndex (có thể không phải nếu parse thất bại hoàn toàn)
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("%s: timestamp index is not DatetimeIndex (got %s)", path.name, type(df.index))
            return None

        # Loại bỏ NaT còn sót trong index (an toàn kép)
        if df.index.isna().any():
            logger.warning("%s: dropped NaT entries from DatetimeIndex", path.name)
            df = df[df.index.notna()]

        df.sort_index(inplace=True)

        # Loại bỏ duplicate timestamps (dữ liệu bị download trùng)
        dup_count = df.index.duplicated().sum()
        if dup_count > 0:
            logger.warning("%s: dropped %d duplicate timestamps", path.name, dup_count)
            df = df[~df.index.duplicated(keep="last")]

        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Loại bỏ row có OHLC không hợp lệ
        before = len(df)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        # Loại bỏ giá âm hoặc bằng 0
        df = df[(df["close"] > 0) & (df["high"] >= df["low"]) & (df["high"] > 0)]
        dropped = before - len(df)
        if dropped > 0:
            logger.warning("%s: dropped %d rows with invalid OHLC values", path.name, dropped)

        if df.empty:
            logger.error("%s: no valid OHLCV data after cleanup", path.name)
            return None

        return df
    except Exception as e:
        logger.error("Cannot load %s: %s", path, e)
        return None


def resample_to_daily(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1H → Daily. Chỉ giữ ngày có đủ ít nhất 20 nến 1H (loại ngày đầu/cuối không đủ).
    Guard: yêu cầu DatetimeIndex không có NaT trước khi resample.
    """
    if not isinstance(df_1h.index, pd.DatetimeIndex):
        raise TypeError(f"resample_to_daily: expected DatetimeIndex, got {type(df_1h.index)}")
    if df_1h.index.isna().any():
        df_1h = df_1h[df_1h.index.notna()]
    df_d = df_1h.resample("D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        bar_count=("close", "count"),
    )
    df_d = df_d[df_d["bar_count"] >= 20].drop(columns=["bar_count"])
    return df_d.dropna(subset=["close"])


# ──────────────────────────────────────────────────────────
# Tính indicators trên daily (có thể dùng cho cả BTC và altcoin)
# ──────────────────────────────────────────────────────────

def compute_daily_indicators(df_d: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm vào daily DataFrame:
      - EMA(20, 50, 200), RSI(14)
      - Accumulation window (rolling 30): acc_high, acc_low, acc_range_pct, acc_bottom
      - Layer 1 pass flag
      - Layer 3 pre-pass flag (không gồm breakout vì breakout dùng giá 1H hiện tại)
    Tất cả đều KHÔNG dùng lookahead: rolling/ewm được tính bình thường trên toàn series;
    khi merge vào 1H sẽ shift 1 ngày để đảm bảo chỉ dùng dữ liệu đã đóng.
    """
    df = df_d.copy()
    close = df["close"]

    for p in EMA_PERIODS:
        df[f"ema{p}"] = _ema(close, p)
    df["rsi"] = _rsi(close, RSI_PERIOD)

    # Accumulation window (30 nến daily cuối, tính rolling trên series hiện tại)
    # Sau khi shift vào merge, row D sẽ represent "dữ liệu tính tới ngày D-1"
    df["acc_high"] = df["high"].rolling(ACCUMULATION_LOOKBACK).max()
    df["acc_low"]  = df["low"].rolling(ACCUMULATION_LOOKBACK).min()
    df["acc_range_pct"] = (
        (df["acc_high"] - df["acc_low"]) / df["acc_low"].replace(0, np.nan) * 100
    )
    df["acc_position"] = (
        df["acc_low"] + (df["acc_high"] - df["acc_low"]) * (ACCUMULATION_POSITION_PCT / 100)
    )

    # Layer 1: price > EMA50 > EMA200, RSI > 40
    df["l1_pass"] = (
        (close > df["ema50"]) &
        (df["ema50"] > df["ema200"]) &
        (df["rsi"] > RSI_MIN_DAILY)
    )

    # Layer 3 pre-conditions (không gồm breakout):
    #   - range < 35%
    #   - yesterday's close >= acc_position (price building pressure near resistance)
    df["l3_range_ok"]      = df["acc_range_pct"] < ACCUMULATION_RANGE_MAX_PCT
    df["l3_prev_close_ok"] = close >= df["acc_position"]
    df["l3_pre_pass"]      = df["l3_range_ok"] & df["l3_prev_close_ok"]

    return df


# ──────────────────────────────────────────────────────────
# Ghép daily indicators vào 1H (không lookahead)
# ──────────────────────────────────────────────────────────

def merge_daily_onto_1h(
    df_1h: pd.DataFrame,
    df_d: pd.DataFrame,
    btc_d: pd.DataFrame | None,
    enable_btc_filter: bool,
) -> pd.DataFrame:
    """
    Ghép daily indicators vào mỗi nến 1H theo nguyên tắc không lookahead:
    - Daily bar ngày D được gán cho tất cả nến 1H của ngày D+1
      (shift index daily lên 1 ngày → merge_asof backward)
    - Điều này tương đương với việc live bot chỉ dùng nến daily đã đóng (iloc[-2])
    """
    # 1H indicators
    df = df_1h.copy()
    df["volume_sma"] = _sma(df["volume"], VOLUME_SMA_PERIOD)
    df["atr_1h"]     = _atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    df["vol_ratio"]  = df["volume"] / df["volume_sma"].replace(0, np.nan)

    # Layer 2 (volume spike): max ratio trong VOLUME_SPIKE_LOOKBACK nến vừa đóng
    # shift(1) → bỏ nến hiện tại; rolling(N).max() → cửa sổ N nến đã đóng
    df["vol_ratio_max"] = (
        df["vol_ratio"].shift(1).rolling(VOLUME_SPIKE_LOOKBACK).max()
    )
    df["l2_pass"] = df["vol_ratio_max"] >= VOLUME_SPIKE_MIN_RATIO

    # Shift daily index lên 1 ngày để ghép đúng vào ngày hôm sau
    daily_cols = ["l1_pass", "l3_pre_pass", "acc_high", "acc_low", "acc_range_pct", "rsi"]
    df_d_shifted = df_d[daily_cols].copy()
    df_d_shifted.index = df_d_shifted.index + pd.Timedelta(days=1)
    df_d_shifted.columns = [c + "_d" for c in daily_cols]

    # merge_asof yêu cầu key đã sort và không có null → drop NaT trước khi merge
    df_reset = df.reset_index()
    df_reset = df_reset[df_reset["timestamp"].notna()].sort_values("timestamp")

    d_shifted_reset = df_d_shifted.reset_index().rename(columns={"index": "timestamp"})
    d_shifted_reset = d_shifted_reset[d_shifted_reset["timestamp"].notna()].sort_values("timestamp")

    merged = pd.merge_asof(
        df_reset,
        d_shifted_reset,
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")

    # BTC filter
    if enable_btc_filter and btc_d is not None:
        btc_cols = btc_d[["l1_pass"]].rename(columns={"l1_pass": "btc_uptrend"}).copy()
        btc_cols.index = btc_cols.index + pd.Timedelta(days=1)
        btc_reset = btc_cols.reset_index().rename(columns={"index": "timestamp"})
        btc_reset = btc_reset[btc_reset["timestamp"].notna()].sort_values("timestamp")
        merged = pd.merge_asof(
            merged.reset_index().sort_values("timestamp"),
            btc_reset,
            on="timestamp",
            direction="backward",
        ).set_index("timestamp")
        merged["btc_uptrend"] = merged["btc_uptrend"].fillna(True)
    else:
        merged["btc_uptrend"] = True

    # Layer 3 breakout: giá hiện tại (1H close) >= 98% của acc_high (từ daily)
    breakout_level = merged["acc_high_d"] * (BREAKOUT_THRESHOLD_PCT / 100)
    merged["l3_pass"] = merged["l3_pre_pass_d"].fillna(False) & (merged["close"] >= breakout_level)

    # Signal: tất cả layers pass
    merged["signal"] = (
        merged["btc_uptrend"].fillna(True) &
        merged["l1_pass_d"].fillna(False) &
        merged["l2_pass"].fillna(False) &
        merged["l3_pass"].fillna(False) &
        merged["acc_high_d"].notna()  # đủ warmup
    )

    return merged


# ──────────────────────────────────────────────────────────
# Tính SL / TP
# ──────────────────────────────────────────────────────────

def compute_sl_tp(entry: float, atr_1h: float, acc_low: float) -> tuple[float, float, float]:
    """
    SL = min(acc_low, entry - ATR_SL_MULTIPLIER × ATR_1h)
    TP1 = entry * 1.10
    TP2 = entry * 1.20
    """
    sl_atr = entry - ATR_SL_MULTIPLIER * atr_1h
    sl = min(float(acc_low), sl_atr) if acc_low > 0 else sl_atr
    if sl >= entry:
        sl = entry - atr_1h
    
    tp1 = entry * 1.10
    tp2 = entry * 1.20
    return sl, tp1, tp2


# ──────────────────────────────────────────────────────────
# Sinh danh sách tín hiệu từ merged DataFrame
# ──────────────────────────────────────────────────────────

def extract_signals(merged: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Lấy tất cả nến 1H có signal=True và tính entry, SL, TP.
    Mỗi row = 1 tín hiệu tiềm năng (chưa dedup).
    """
    sig_rows = merged[merged["signal"]].copy()
    if sig_rows.empty:
        return pd.DataFrame()

    records = []
    for ts, row in sig_rows.iterrows():
        # Bỏ qua nếu timestamp index là NaT (do merge tạo ra)
        if pd.isna(ts):
            continue
        entry = float(row["close"])
        atr_val = float(row["atr_1h"]) if not pd.isna(row["atr_1h"]) else entry * 0.01
        acc_low = float(row["acc_low_d"]) if not pd.isna(row["acc_low_d"]) else 0.0
        sl, tp1, tp2 = compute_sl_tp(entry, atr_val, acc_low)
        risk_dist = entry - sl
        if risk_dist / entry * 100 > MAX_RISK_PER_TRADE_PCT:
            continue

        records.append({
            "symbol":           symbol,
            "signal_time":      ts,
            "entry":            round(entry, 8),
            "stop_loss":        round(sl, 8),
            "tp1":              round(tp1, 8),
            "tp2":              round(tp2, 8),
            "risk_pct":         round(risk_dist / entry * 100, 3) if entry > 0 else 0,
            "rr_ratio":         round((tp1 - entry) / risk_dist, 2) if risk_dist > 0 else 0,
            "volume_ratio":     round(float(row.get("vol_ratio_max", 0) or 0), 2),
            "acc_range_pct":    round(float(row.get("acc_range_pct_d", 0) or 0), 2),
            "rsi_daily":        round(float(row.get("rsi_d", 0) or 0), 1),
            "atr_1h_pct":       round(atr_val / entry * 100, 3) if entry > 0 else 0,
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────
# Mô phỏng trades (walk-forward trên nến 1H)
# ──────────────────────────────────────────────────────────

def simulate_trades(
    df_1h: pd.DataFrame,
    signals: pd.DataFrame,
    timeout_days: int = TRADE_TIMEOUT_DAYS,
) -> pd.DataFrame:
    """
    Với mỗi tín hiệu:
    - Enter tại OPEN của nến 1H TIẾP THEO sau signal
    - Kiểm tra SL/TP trên từng nến 1H tiếp theo (dùng high/low)
    - Nếu sau timeout_days ngày vẫn không chạm → đóng tại close cuối
    - Dedup: bỏ qua tín hiệu mới nếu lệnh trước chưa đóng (cùng symbol)

    Quy tắc intrabar (bảo thủ):
      - Nếu cùng 1 nến có low ≤ SL và high ≥ TP1 → SL được ưu tiên (LOSS)
      - Nếu đã hit TP1 trước đó và nến này low ≤ SL → WIN_PARTIAL (50% vào TP1)

    Kết quả mỗi lệnh:
      WIN_FULL    : TP2 chạm (2 phần đều thắng)
      WIN_PARTIAL : TP1 chạm, TP2 không chạm (sau đó SL hoặc timeout)
      LOSS        : SL chạm trước TP1
      TIMEOUT     : hết thời gian, chưa chạm SL hay TP1
    """
    if signals.empty or df_1h.empty:
        return pd.DataFrame()

    timeout_bars = timeout_days * 24
    df_idx = df_1h.index
    idx_map = {ts: i for i, ts in enumerate(df_idx)}

    results = []
    last_exit_bar = -1  # dedup: bỏ qua signal nếu lệnh trước chưa đóng

    for _, sig in signals.sort_values("signal_time").iterrows():
        signal_ts = sig["signal_time"]

        # Bỏ qua signal có timestamp không hợp lệ
        if pd.isna(signal_ts):
            continue

        # Tìm vị trí của nến signal trong df_1h
        if signal_ts not in idx_map:
            # Tìm bar gần nhất sau signal_ts
            pos_arr = df_idx.searchsorted(signal_ts)
            if pos_arr >= len(df_idx):
                continue
            signal_bar = int(pos_arr)
        else:
            signal_bar = idx_map[signal_ts]

        entry_bar = signal_bar + 1  # enter tại open nến tiếp theo
        if entry_bar >= len(df_1h):
            continue

        # Dedup: bỏ qua nếu lệnh trước chưa đóng
        if entry_bar <= last_exit_bar:
            continue

        entry_price = float(df_1h.iloc[entry_bar]["open"])
        if entry_price <= 0:
            continue

        # Tính lại SL/TP dựa trên giá vào thực tế
        sl   = float(sig["stop_loss"])
        tp1  = entry_price * 1.10
        tp2  = entry_price * 1.20
        risk_dist = entry_price - sl
        if risk_dist <= 0:
            continue

        outcome    = "TIMEOUT"
        exit_price = None
        exit_bar   = entry_bar
        tp1_hit    = False

        max_bar = min(entry_bar + timeout_bars, len(df_1h))

        for j in range(entry_bar, max_bar):
            bar     = df_1h.iloc[j]
            bar_low  = float(bar["low"])
            bar_high = float(bar["high"])

            # SL check (bảo thủ: ưu tiên SL nếu cùng nến)
            if bar_low <= sl:
                outcome    = "WIN_PARTIAL" if tp1_hit else "LOSS"
                exit_price = sl
                exit_bar   = j
                break

            # TP1 check
            if not tp1_hit and bar_high >= tp1:
                tp1_hit = True
                # Tiếp tục giữ để chờ TP2

            # TP2 check
            if tp1_hit and bar_high >= tp2:
                outcome    = "WIN_FULL"
                exit_price = tp2
                exit_bar   = j
                break

        # TIMEOUT hoặc cuối dữ liệu
        if exit_price is None:
            exit_bar   = min(max_bar - 1, len(df_1h) - 1)
            exit_price = float(df_1h.iloc[exit_bar]["close"])
            outcome    = "WIN_PARTIAL" if tp1_hit else "TIMEOUT"

        last_exit_bar = exit_bar

        # Tính P&L (% từ entry, chiến lược 50% tại TP1 + 50% tại TP2)
        if outcome == "WIN_FULL":
            pnl_pct = (((tp1 - entry_price) + (tp2 - entry_price)) / entry_price) * 50
        elif outcome == "WIN_PARTIAL":
            if tp1_hit:
                # 50% đóng tại TP1, 50% đóng tại exit (SL hoặc timeout)
                half2_pnl = (exit_price - entry_price) / entry_price * 100
                pnl_pct   = 0.5 * ((tp1 - entry_price) / entry_price * 100) + 0.5 * half2_pnl
            else:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
        elif outcome == "LOSS":
            pnl_pct = -risk_dist / entry_price * 100
        else:  # TIMEOUT
            pnl_pct = (exit_price - entry_price) / entry_price * 100

        bars_held = exit_bar - entry_bar

        results.append({
            "symbol":        sig["symbol"],
            "signal_time":   signal_ts,
            "entry_time":    df_idx[entry_bar],
            "exit_time":     df_idx[exit_bar],
            "entry":         round(entry_price, 8),
            "stop_loss":     round(sl, 8),
            "tp1":           round(tp1, 8),
            "tp2":           round(tp2, 8),
            "exit_price":    round(exit_price, 8),
            "outcome":       outcome,
            "pnl_pct":       round(pnl_pct, 3),
            "bars_held":     bars_held,
            "days_held":     round(bars_held / 24, 1),
            "risk_pct":      round(sig["risk_pct"], 3),
            "rr_ratio":      round(sig["rr_ratio"], 2),
            "volume_ratio":  round(sig["volume_ratio"], 2),
            "acc_range_pct": round(sig["acc_range_pct"], 2),
            "rsi_daily":     round(sig["rsi_daily"], 1),
        })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────
# Tính metrics
# ──────────────────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame) -> dict:
    """Tính các chỉ số hiệu suất từ danh sách trades."""
    if trades.empty:
        return {"total_trades": 0}

    total   = len(trades)
    wins    = trades[trades["outcome"].isin(["WIN_FULL", "WIN_PARTIAL"])]
    losses  = trades[trades["outcome"] == "LOSS"]
    timeout = trades[trades["outcome"] == "TIMEOUT"]

    gross_profit = wins["pnl_pct"].sum() if len(wins) > 0 else 0.0
    gross_loss   = losses["pnl_pct"].sum() if len(losses) > 0 else 0.0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    avg_win  = wins["pnl_pct"].mean()   if len(wins)   > 0 else 0.0
    avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0.0

    # Equity curve (compounded, fixed 1% risk per trade)
    equity   = (1 + trades["pnl_pct"] / 100).cumprod()
    total_return_pct = (equity.iloc[-1] - 1) * 100

    roll_max     = equity.cummax()
    drawdown     = (equity - roll_max) / roll_max
    max_dd_pct   = drawdown.min() * 100

    win_rate = len(wins) / total * 100

    return {
        "total_trades":      total,
        "wins":              len(wins),
        "losses":            len(losses),
        "timeouts":          len(timeout),
        "win_rate_pct":      round(win_rate, 1),
        "profit_factor":     round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
        "avg_win_pct":       round(avg_win, 2),
        "avg_loss_pct":      round(avg_loss, 2),
        "total_return_pct":  round(total_return_pct, 2),
        "max_drawdown_pct":  round(max_dd_pct, 2),
        "gross_profit_pct":  round(gross_profit, 2),
        "gross_loss_pct":    round(gross_loss, 2),
        "avg_days_held":     round(trades["days_held"].mean(), 1),
        "avg_rr_signal":     round(trades["rr_ratio"].mean(), 2),
    }


def build_equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    """Tính equity curve theo thời gian (entry_time làm mốc)."""
    if trades.empty:
        return pd.DataFrame(columns=["time", "equity"])
    df = trades.sort_values("entry_time")[["entry_time", "pnl_pct"]].copy()
    df["equity"] = (1 + df["pnl_pct"] / 100).cumprod() * 100
    return df.rename(columns={"entry_time": "time"})


# ──────────────────────────────────────────────────────────
# In kết quả
# ──────────────────────────────────────────────────────────

def print_metrics(symbol: str, m: dict) -> None:
    if m.get("total_trades", 0) == 0:
        logger.info("%-12s │ No trades", symbol)
        return
    logger.info(
        "%-12s │ Trades: %3d  WR: %5.1f%%  PF: %5.2f  Return: %+7.2f%%  MaxDD: %6.2f%%  "
        "AvgWin: %+5.2f%%  AvgLoss: %+5.2f%%  AvgDays: %.1f",
        symbol,
        m["total_trades"],
        m["win_rate_pct"],
        m["profit_factor"],
        m["total_return_pct"],
        m["max_drawdown_pct"],
        m["avg_win_pct"],
        m["avg_loss_pct"],
        m["avg_days_held"],
    )


# ──────────────────────────────────────────────────────────
# Per-symbol backtest pipeline
# ──────────────────────────────────────────────────────────

def run_backtest_for_symbol(
    symbol:          str,
    data_dir:        Path,
    btc_d_indicators: pd.DataFrame | None,
    enable_btc_filter: bool,
    start_date:      pd.Timestamp,
    end_date:        pd.Timestamp,
    timeout_days:    int,
) -> tuple[pd.DataFrame, dict]:
    """
    Chạy toàn bộ pipeline backtest cho một symbol.
    Trả về (trades_df, metrics_dict).
    """
    # 1. Load 1H data
    df_1h = load_ohlcv(data_dir, symbol, "1h")
    if df_1h is None or len(df_1h) < VOLUME_SMA_PERIOD + 10:
        logger.warning("%-12s │ Insufficient 1H data, skip", symbol)
        return pd.DataFrame(), {}

    # 2. Load hoặc resample daily data
    df_d = load_ohlcv(data_dir, symbol, "1d")
    if df_d is None:
        df_d = resample_to_daily(df_1h)
        logger.debug("%-12s │ Resampled 1H→1D (%d daily bars)", symbol, len(df_d))

    if len(df_d) < WARMUP_DAILY_BARS:
        logger.warning("%-12s │ Only %d daily bars (need %d), skip", symbol, len(df_d), WARMUP_DAILY_BARS)
        return pd.DataFrame(), {}

    # 3. Lọc theo khoảng thời gian backtest
    df_1h = df_1h[start_date: end_date]
    df_d  = df_d[start_date - pd.Timedelta(days=WARMUP_DAILY_BARS + 5): end_date]

    if df_1h.empty:
        logger.warning("%-12s │ No 1H data in date range, skip", symbol)
        return pd.DataFrame(), {}

    # 4. Tính indicators daily
    df_d_ind = compute_daily_indicators(df_d)

    # 5. Merge daily → 1H, tính signals
    merged = merge_daily_onto_1h(df_1h, df_d_ind, btc_d_indicators, enable_btc_filter)

    # 6. Trích xuất tín hiệu
    signals = extract_signals(merged, symbol)
    if signals.empty:
        logger.info("%-12s │ 0 signals found in period", symbol)
        return pd.DataFrame(), {"total_trades": 0}

    logger.debug("%-12s │ %d raw signals found", symbol, len(signals))

    # 7. Mô phỏng trades
    trades = simulate_trades(df_1h, signals, timeout_days=timeout_days)

    # 8. Metrics
    metrics = compute_metrics(trades)
    print_metrics(symbol, metrics)

    return trades, metrics


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest Binance Spot signal bot (3-layer filter)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "binance-data-downloader" / "data" / "processed",
        help="Thư mục chứa CSV từ binance-data-downloader (default: ../binance-data-downloader/data/processed)",
    )
    p.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Danh sách symbol (e.g. BTCUSDT ETHUSDT). Mặc định: tự detect từ files *_1h.csv",
    )
    p.add_argument(
        "--start",
        default="2021-01-01",
        help="Ngày bắt đầu backtest (YYYY-MM-DD, default: 2021-01-01)",
    )
    p.add_argument(
        "--end",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Ngày kết thúc backtest (YYYY-MM-DD, default: hôm nay)",
    )
    p.add_argument(
        "--no-btc-filter",
        action="store_true",
        help="Tắt bộ lọc xu hướng BTC",
    )
    p.add_argument(
        "--timeout-days",
        type=int,
        default=TRADE_TIMEOUT_DAYS,
        help=f"Đóng lệnh sau N ngày nếu không chạm SL/TP (default: {TRADE_TIMEOUT_DAYS})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "backtest_results",
        help="Thư mục lưu kết quả (default: ./backtest_results)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir
    enable_btc_filter: bool = not args.no_btc_filter

    start_date = pd.Timestamp(args.start)
    end_date   = pd.Timestamp(args.end) + pd.Timedelta(days=1)

    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        logger.error("Hãy chạy binance-data-downloader trước để tải dữ liệu.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect symbols từ *_1h.csv nếu không truyền --symbols
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        csv_files = sorted(data_dir.glob("*_1h.csv"))
        symbols   = [f.stem.replace("_1h", "") for f in csv_files]
        if not symbols:
            logger.error("Không tìm thấy file *_1h.csv trong %s", data_dir)
            sys.exit(1)
        logger.info("Auto-detected %d symbols: %s%s",
                    len(symbols), ", ".join(symbols[:10]),
                    f" ... +{len(symbols)-10} more" if len(symbols) > 10 else "")

    logger.info("=" * 70)
    logger.info("BACKTEST  │  Period: %s → %s  │  Symbols: %d  │  BTC filter: %s",
                args.start, args.end, len(symbols), "ON" if enable_btc_filter else "OFF")
    logger.info("=" * 70)

    # Load BTC daily cho bộ lọc thị trường (một lần)
    btc_d_indicators = None
    if enable_btc_filter:
        btc_1h = load_ohlcv(data_dir, BTC_SYMBOL, "1h")
        btc_1d = load_ohlcv(data_dir, BTC_SYMBOL, "1d")
        if btc_1h is not None:
            btc_raw = btc_1d if btc_1d is not None else resample_to_daily(btc_1h)
            if len(btc_raw) >= WARMUP_DAILY_BARS:
                btc_d_indicators = compute_daily_indicators(btc_raw)
                logger.info("BTC daily data loaded: %d bars", len(btc_d_indicators))
            else:
                logger.warning("Insufficient BTC data (%d bars), BTC filter disabled", len(btc_raw))
                enable_btc_filter = False
        else:
            logger.warning("BTC 1H data not found, BTC filter disabled")
            enable_btc_filter = False

    # Chạy backtest cho từng symbol
    all_trades_list: list[pd.DataFrame] = []
    all_metrics: dict[str, dict] = {}

    for sym in symbols:
        if sym == BTC_SYMBOL and enable_btc_filter:
            # BTC vẫn được backtest, chỉ không dùng BTC filter cho chính nó
            trades, metrics = run_backtest_for_symbol(
                sym, data_dir, btc_d_indicators, False, start_date, end_date, args.timeout_days
            )
        else:
            trades, metrics = run_backtest_for_symbol(
                sym, data_dir, btc_d_indicators, enable_btc_filter, start_date, end_date, args.timeout_days
            )

        if not trades.empty:
            all_trades_list.append(trades)
            # Lưu per-symbol
            sym_path = output_dir / f"{sym}_trades.csv"
            trades.to_csv(sym_path, index=False)

        all_metrics[sym] = metrics

    # Gộp tất cả trades
    if not all_trades_list:
        logger.warning("Không có lệnh nào được ghi nhận trong toàn bộ backtest.")
        return

    all_trades = pd.concat(all_trades_list, ignore_index=True).sort_values("entry_time")
    all_trades_path = output_dir / "all_trades.csv"
    all_trades.to_csv(all_trades_path, index=False)
    logger.info("Saved %d total trades → %s", len(all_trades), all_trades_path)

    # Portfolio metrics
    portfolio_metrics = compute_metrics(all_trades)

    # Summary table
    logger.info("")
    logger.info("=" * 70)
    logger.info("PER-SYMBOL SUMMARY")
    logger.info("=" * 70)
    logger.info(
        "%-12s  %6s  %6s  %6s  %8s  %8s  %7s  %7s",
        "Symbol", "Trades", "WinRate", "PF", "Return%", "MaxDD%", "AvgWin", "AvgLoss"
    )
    logger.info("-" * 70)
    for sym, m in all_metrics.items():
        if m.get("total_trades", 0) == 0:
            continue
        logger.info(
            "%-12s  %6d  %5.1f%%  %6.2f  %+7.2f%%  %7.2f%%  %+6.2f%%  %+6.2f%%",
            sym,
            m["total_trades"],
            m["win_rate_pct"],
            m["profit_factor"],
            m["total_return_pct"],
            m["max_drawdown_pct"],
            m["avg_win_pct"],
            m["avg_loss_pct"],
        )
    logger.info("=" * 70)
    logger.info("PORTFOLIO TOTAL (%d symbols, %d trades)", len(all_trades_list), portfolio_metrics["total_trades"])
    logger.info("  Win Rate:       %5.1f%%  (W:%d / L:%d / T:%d)",
                portfolio_metrics["win_rate_pct"],
                portfolio_metrics["wins"],
                portfolio_metrics["losses"],
                portfolio_metrics["timeouts"])
    logger.info("  Profit Factor:  %5.2f", portfolio_metrics["profit_factor"])
    logger.info("  Total Return:   %+.2f%%", portfolio_metrics["total_return_pct"])
    logger.info("  Max Drawdown:   %.2f%%", portfolio_metrics["max_drawdown_pct"])
    logger.info("  Avg Win:        %+.2f%%  |  Avg Loss: %.2f%%",
                portfolio_metrics["avg_win_pct"], portfolio_metrics["avg_loss_pct"])
    logger.info("  Avg Days Held:  %.1f", portfolio_metrics["avg_days_held"])
    logger.info("=" * 70)

    # Equity curve
    equity_df = build_equity_curve(all_trades)
    eq_path = output_dir / "equity_curve.csv"
    equity_df.to_csv(eq_path, index=False)

    # Portfolio summary CSV
    summary_rows = [{"symbol": sym, **m} for sym, m in all_metrics.items() if m.get("total_trades", 0) > 0]
    summary_rows.append({"symbol": "_PORTFOLIO_", **portfolio_metrics})
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)

    logger.info("")
    logger.info("Output files:")
    logger.info("  %s  (all trades)", all_trades_path)
    logger.info("  %s  (equity curve)", eq_path)
    logger.info("  %s  (summary)", output_dir / "summary.csv")
    logger.info("  %s/*.csv  (per-symbol trades)", output_dir)


if __name__ == "__main__":
    main()
