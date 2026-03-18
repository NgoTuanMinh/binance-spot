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
import json
import logging
import random
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
VOLUME_SPIKE_MIN_RATIO      = 0.6
VOLUME_SPIKE_LOOKBACK       = 2       # số nến 1H đã đóng để kiểm tra spike
ACCUMULATION_LOOKBACK       = 30      # nến daily
ACCUMULATION_RANGE_MAX_PCT  = 50.0
ACCUMULATION_POSITION_PCT   = 30.0
BREAKOUT_THRESHOLD_PCT      = 92.0
ATR_SL_MULTIPLIER           = 0.4
RSI_MIN_DAILY               = 28
TP1_RR_RATIO                = 1.5
TP2_RR_RATIO                = 12.0
BTC_SYMBOL                  = "BTCUSDT"

# Tham số backtest (profile tối ưu ~536% return, DD ~60%)
TRADE_TIMEOUT_DAYS          = 60      # đóng lệnh sau N ngày nếu không chạm SL/TP
RISK_PER_TRADE_PCT          = 1.0     # % equity rủi ro mỗi lệnh (để tính equity curve)
WARMUP_DAILY_BARS           = 250     # số nến daily tối thiểu trước khi bắt đầu tín hiệu
MAX_RISK_PER_TRADE_PCT      = 25.0     # maximum risk per trade in percentage
MAX_OPEN_POSITIONS          = 15       # tối đa số lệnh mở đồng thời
ALLOCATION_PER_TRADE_PCT    = 25.0     # phân bổ vốn mỗi lệnh (% equity tại thời điểm vào)
INITIAL_EQUITY              = 10_000.0 # vốn giả lập ban đầu (USDT)
BACKTEST_ENABLE_BTC_FILTER_DEFAULT = False

# Mô hình chi phí giao dịch (thực tế hơn)
TAKER_FEE_BPS               = 10.0      # Binance spot taker mặc định ~0.1%
SLIPPAGE_BPS                = 8.0       # trượt giá tĩnh cho mỗi lần khớp lệnh
SLIPPAGE_ATR_MULT           = 0.0       # cộng thêm slippage theo ATR% (0 = tắt)
MAX_TOTAL_SLIPPAGE_BPS      = 80.0      # trần slippage để tránh giá trị bất thường

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
        # Đọc timestamp dạng raw string trước để xử lý các file có trộn format
        # (vd: vừa có "YYYY-MM-DD HH:MM:SS.000" vừa có "YYYY-MM-DD HH:MM:SS").
        df = pd.read_csv(path)

        # Kiểm tra timestamp có parse thành datetime không
        if "timestamp" not in df.columns:
            logger.error("Missing 'timestamp' column in %s", path)
            return None

        # pandas>=2.0 có thể suy luận format quá chặt nếu file bị trộn format.
        # Ưu tiên parse "mixed", fallback về parse mặc định nếu version cũ không hỗ trợ.
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False, format="mixed")
        except TypeError:
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
    TP1 = entry + risk_dist * TP1_RR_RATIO
    TP2 = entry + risk_dist * TP2_RR_RATIO
    """
    sl_atr = entry - ATR_SL_MULTIPLIER * atr_1h
    sl = min(float(acc_low), sl_atr) if acc_low > 0 else sl_atr
    if sl >= entry:
        sl = entry - atr_1h

    risk_dist = max(entry - sl, 0.0)
    tp1 = entry + risk_dist * TP1_RR_RATIO
    tp2 = entry + risk_dist * TP2_RR_RATIO
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
# Trading cost helpers (fee + slippage)
# ──────────────────────────────────────────────────────────
def _effective_slippage_rate(
    atr_1h_pct: float,
    slippage_bps: float,
    slippage_atr_mult: float,
    max_total_slippage_bps: float,
) -> float:
    """
    Tính slippage rate theo:
      base_slippage + slippage_atr_mult * ATR%
    Ví dụ:
      base=8bps, ATR%=1.2, mult=0.2 -> total ~= 8bps + 24bps = 32bps
    """
    base = max(slippage_bps, 0.0) / 10_000
    dyn = max(slippage_atr_mult, 0.0) * max(atr_1h_pct, 0.0) / 100
    cap = max(max_total_slippage_bps, 0.0) / 10_000
    return min(base + dyn, cap)


def _net_leg_return_pct(
    entry_px: float,
    exit_px: float,
    fee_rate: float,
) -> float:
    """Return % cho 1 leg sau khi trừ phí mua + bán."""
    if entry_px <= 0:
        return 0.0
    entry_cost = entry_px * (1 + fee_rate)
    exit_proceeds = exit_px * (1 - fee_rate)
    return (exit_proceeds / entry_cost - 1) * 100


# ──────────────────────────────────────────────────────────
# Mô phỏng trades (walk-forward trên nến 1H)
# ──────────────────────────────────────────────────────────

def simulate_trades(
    df_1h: pd.DataFrame,
    signals: pd.DataFrame,
    timeout_days: int = TRADE_TIMEOUT_DAYS,
    taker_fee_bps: float = TAKER_FEE_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    slippage_atr_mult: float = SLIPPAGE_ATR_MULT,
    max_total_slippage_bps: float = MAX_TOTAL_SLIPPAGE_BPS,
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

    fee_rate = max(taker_fee_bps, 0.0) / 10_000
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

        atr_sig_pct = float(sig.get("atr_1h_pct", 0.0) or 0.0)
        slip_rate = _effective_slippage_rate(
            atr_1h_pct=atr_sig_pct,
            slippage_bps=slippage_bps,
            slippage_atr_mult=slippage_atr_mult,
            max_total_slippage_bps=max_total_slippage_bps,
        )

        # BUY bị trượt giá bất lợi: mua cao hơn giá lý thuyết
        entry_exec = entry_price * (1 + slip_rate)

        # Tính lại SL/TP dựa trên giá vào thực tế
        sl   = float(sig["stop_loss"])
        risk_dist = entry_price - sl
        if risk_dist <= 0:
            continue
        tp1  = entry_price + risk_dist * TP1_RR_RATIO
        tp2  = entry_price + risk_dist * TP2_RR_RATIO

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

        # SELL cũng chịu slippage bất lợi: bán thấp hơn giá lý thuyết
        exit_exec = exit_price * (1 - slip_rate)
        tp1_exec = tp1 * (1 - slip_rate)
        tp2_exec = tp2 * (1 - slip_rate)

        # Tính P&L (% từ entry) sau phí + slippage
        # chiến lược scale-out 50% tại TP1 + 50% tại TP2/SL/timeout
        if outcome == "WIN_FULL":
            leg1 = _net_leg_return_pct(entry_exec, tp1_exec, fee_rate)
            leg2 = _net_leg_return_pct(entry_exec, tp2_exec, fee_rate)
            pnl_pct = 0.5 * leg1 + 0.5 * leg2
        elif outcome == "WIN_PARTIAL":
            if tp1_hit:
                # 50% đóng tại TP1, 50% đóng tại exit (SL hoặc timeout)
                leg1 = _net_leg_return_pct(entry_exec, tp1_exec, fee_rate)
                leg2 = _net_leg_return_pct(entry_exec, exit_exec, fee_rate)
                pnl_pct = 0.5 * leg1 + 0.5 * leg2
            else:
                pnl_pct = _net_leg_return_pct(entry_exec, exit_exec, fee_rate)
        else:  # TIMEOUT
            pnl_pct = _net_leg_return_pct(entry_exec, exit_exec, fee_rate)

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
            "fee_bps":       round(taker_fee_bps, 3),
            "slippage_bps_effective": round(slip_rate * 10_000, 3),
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
    wins    = trades[trades["pnl_pct"] > 0]
    losses  = trades[trades["pnl_pct"] < 0]
    timeout = trades[trades["outcome"] == "TIMEOUT"]
    loss_count_by_outcome = len(trades[trades["outcome"] == "LOSS"])

    gross_profit = wins["pnl_pct"].sum() if not wins.empty else 0.0
    gross_loss   = losses["pnl_pct"].sum() if not losses.empty else 0.0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    avg_win  = wins["pnl_pct"].mean()   if not wins.empty else 0.0
    avg_loss = losses["pnl_pct"].mean() if not losses.empty else 0.0

    # Equity curve (compounded, fixed 1% risk per trade)
    equity   = (1 + trades["pnl_pct"] / 100).cumprod()
    total_return_pct = (equity.iloc[-1] - 1) * 100

    roll_max     = equity.cummax()
    drawdown     = (equity - roll_max) / roll_max
    max_dd_pct   = drawdown.min() * 100

    win_rate = len(wins) / total * 100

    return {
        "total_trades":      total,
        "wins":              int(len(wins)),
        "losses":            int(len(losses)),
        "losses_outcome_only": int(loss_count_by_outcome),
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


def apply_portfolio_allocation(
    trades: pd.DataFrame,
    initial_equity: float,
    allocation_per_trade_pct: float,
    max_open_positions: int,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Mô phỏng quản trị vốn thực chiến:
    - Mỗi lệnh dùng allocation_per_trade_pct % equity tại thời điểm vào.
    - Tối đa max_open_positions lệnh mở đồng thời.
    - Lệnh vượt giới hạn sẽ bị skip.
    - Không mark-to-market intrabar cho lệnh mở; equity cập nhật khi lệnh đóng.
    """
    if trades.empty:
        return pd.DataFrame(), {"total_trades": 0}, pd.DataFrame(columns=["time", "equity"])

    alloc_rate = max(allocation_per_trade_pct, 0.0) / 100
    max_open = max(int(max_open_positions), 1)
    equity0 = float(max(initial_equity, 0.0))

    df = trades.sort_values("entry_time").copy()
    cash = equity0
    open_positions: list[dict] = []
    executed_rows: list[dict] = []

    skipped_capacity = 0
    skipped_cash = 0
    equity_points: list[dict] = [{"time": df["entry_time"].min(), "equity": equity0}]

    def _release_positions(until_time: pd.Timestamp) -> None:
        nonlocal cash, open_positions
        matured = [p for p in open_positions if p["exit_time"] <= until_time]
        if not matured:
            return
        for pos in sorted(matured, key=lambda x: x["exit_time"]):
            cash += pos["notional"] * (1 + pos["pnl_pct"] / 100)
            equity_points.append({"time": pos["exit_time"], "equity": cash + sum(x["notional"] for x in open_positions if x is not pos)})
        open_positions = [p for p in open_positions if p["exit_time"] > until_time]

    for _, row in df.iterrows():
        entry_time = row["entry_time"]
        _release_positions(entry_time)

        if len(open_positions) >= max_open:
            skipped_capacity += 1
            continue

        equity_mark = cash + sum(p["notional"] for p in open_positions)
        notional = equity_mark * alloc_rate
        if notional <= 0 or cash < notional:
            skipped_cash += 1
            continue

        cash -= notional
        pos = {
            "exit_time": row["exit_time"],
            "pnl_pct": float(row["pnl_pct"]),
            "notional": float(notional),
        }
        open_positions.append(pos)

        row_dict = row.to_dict()
        row_dict["allocated_notional"] = round(notional, 4)
        row_dict["equity_at_entry"] = round(equity_mark, 4)
        row_dict["cash_after_entry"] = round(cash, 4)
        row_dict["pnl_usdt"] = round(notional * float(row["pnl_pct"]) / 100, 4)
        executed_rows.append(row_dict)

    if open_positions:
        for pos in sorted(open_positions, key=lambda x: x["exit_time"]):
            cash += pos["notional"] * (1 + pos["pnl_pct"] / 100)
            equity_points.append({"time": pos["exit_time"], "equity": cash})
        open_positions = []

    executed = pd.DataFrame(executed_rows).sort_values("entry_time") if executed_rows else pd.DataFrame()
    eq_df = pd.DataFrame(equity_points).sort_values("time").drop_duplicates(subset=["time"], keep="last")
    if eq_df.empty:
        eq_df = pd.DataFrame([{"time": df["entry_time"].min(), "equity": equity0}])
    eq_df["roll_max"] = eq_df["equity"].cummax()
    eq_df["drawdown"] = (eq_df["equity"] - eq_df["roll_max"]) / eq_df["roll_max"]

    if executed.empty:
        return executed, {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "total_return_pct": round((float(eq_df.iloc[-1]["equity"]) / equity0 - 1) * 100, 2) if equity0 > 0 else 0.0,
            "max_drawdown_pct": round(float(eq_df["drawdown"].min() * 100), 2) if "drawdown" in eq_df else 0.0,
            "gross_profit_pct": 0.0,
            "gross_loss_pct": 0.0,
            "avg_days_held": 0.0,
            "avg_rr_signal": 0.0,
            "skipped_capacity": skipped_capacity,
            "skipped_cash": skipped_cash,
            "final_equity": round(float(eq_df.iloc[-1]["equity"]), 4),
            "initial_equity": round(equity0, 4),
            "allocation_per_trade_pct": allocation_per_trade_pct,
            "max_open_positions": max_open,
        }, eq_df[["time", "equity"]]

    wins = executed[executed["pnl_usdt"] > 0]
    losses = executed[executed["pnl_usdt"] < 0]
    gross_profit = wins["pnl_usdt"].sum() if not wins.empty else 0.0
    gross_loss = losses["pnl_usdt"].sum() if not losses.empty else 0.0
    pf = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    final_equity = float(eq_df.iloc[-1]["equity"])
    total_return_pct = (final_equity / equity0 - 1) * 100 if equity0 > 0 else 0.0
    max_dd_pct = float(eq_df["drawdown"].min() * 100)

    metrics = {
        "total_trades": int(len(executed)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "timeouts": int(len(executed[executed["outcome"] == "TIMEOUT"])),
        "win_rate_pct": round(len(wins) / len(executed) * 100, 1),
        "profit_factor": round(pf, 2) if pf != float("inf") else 999.0,
        "avg_win_pct": round(wins["pnl_pct"].mean(), 2) if not wins.empty else 0.0,
        "avg_loss_pct": round(losses["pnl_pct"].mean(), 2) if not losses.empty else 0.0,
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "gross_profit_pct": round(gross_profit / equity0 * 100, 2) if equity0 > 0 else 0.0,
        "gross_loss_pct": round(gross_loss / equity0 * 100, 2) if equity0 > 0 else 0.0,
        "avg_days_held": round(executed["days_held"].mean(), 1),
        "avg_rr_signal": round(executed["rr_ratio"].mean(), 2),
        "final_equity": round(final_equity, 4),
        "initial_equity": round(equity0, 4),
        "skipped_capacity": skipped_capacity,
        "skipped_cash": skipped_cash,
        "allocation_per_trade_pct": allocation_per_trade_pct,
        "max_open_positions": max_open,
    }
    return executed, metrics, eq_df[["time", "equity"]]


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
    taker_fee_bps:   float,
    slippage_bps:    float,
    slippage_atr_mult: float,
    max_total_slippage_bps: float,
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
    trades = simulate_trades(
        df_1h,
        signals,
        timeout_days=timeout_days,
        taker_fee_bps=taker_fee_bps,
        slippage_bps=slippage_bps,
        slippage_atr_mult=slippage_atr_mult,
        max_total_slippage_bps=max_total_slippage_bps,
    )

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
        "--btc-filter",
        action="store_true",
        help=f"Bật bộ lọc xu hướng BTC (default: {'ON' if BACKTEST_ENABLE_BTC_FILTER_DEFAULT else 'OFF'})",
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
    p.add_argument(
        "--profile",
        choices=["default", "aggressive"],
        default="default",
        help="Profile tham số: default (~536%% return) | aggressive (tối đa return, DD sâu hơn)",
    )
    p.add_argument("--volume-spike-min-ratio", type=float, default=None, help=f"Override VOLUME_SPIKE_MIN_RATIO (default: {VOLUME_SPIKE_MIN_RATIO})")
    p.add_argument("--acc-range-max-pct", type=float, default=None, help=f"Override ACCUMULATION_RANGE_MAX_PCT (default: {ACCUMULATION_RANGE_MAX_PCT})")
    p.add_argument("--breakout-threshold-pct", type=float, default=None, help=f"Override BREAKOUT_THRESHOLD_PCT (default: {BREAKOUT_THRESHOLD_PCT})")
    p.add_argument("--atr-sl-multiplier", type=float, default=None, help=f"Override ATR_SL_MULTIPLIER (default: {ATR_SL_MULTIPLIER})")
    p.add_argument("--max-risk-per-trade-pct", type=float, default=None, help=f"Override MAX_RISK_PER_TRADE_PCT (default: {MAX_RISK_PER_TRADE_PCT})")
    p.add_argument("--rsi-min-daily", type=float, default=None, help=f"Override RSI_MIN_DAILY (default: {RSI_MIN_DAILY})")
    p.add_argument("--tp1-rr-ratio", type=float, default=None, help=f"Override TP1_RR_RATIO (default: {TP1_RR_RATIO})")
    p.add_argument("--tp2-rr-ratio", type=float, default=None, help=f"Override TP2_RR_RATIO (default: {TP2_RR_RATIO})")

    # Trading cost model
    p.add_argument("--taker-fee-bps", type=float, default=TAKER_FEE_BPS, help=f"Taker fee in bps (default: {TAKER_FEE_BPS})")
    p.add_argument("--slippage-bps", type=float, default=SLIPPAGE_BPS, help=f"Base slippage per side in bps (default: {SLIPPAGE_BPS})")
    p.add_argument("--slippage-atr-mult", type=float, default=SLIPPAGE_ATR_MULT, help=f"Dynamic slippage multiplier by ATR%% (default: {SLIPPAGE_ATR_MULT})")
    p.add_argument("--max-total-slippage-bps", type=float, default=MAX_TOTAL_SLIPPAGE_BPS, help=f"Cap total slippage in bps (default: {MAX_TOTAL_SLIPPAGE_BPS})")

    # Walk-forward optimization
    p.add_argument("--walk-forward", action="store_true", help="Run walk-forward optimization mode")
    p.add_argument("--wf-train-months", type=int, default=24, help="Train window size (months)")
    p.add_argument("--wf-test-months", type=int, default=3, help="Test window size (months)")
    p.add_argument("--wf-step-months", type=int, default=3, help="Window step size (months)")
    p.add_argument("--wf-max-combos", type=int, default=40, help="Max parameter combinations to evaluate")
    p.add_argument("--wf-seed", type=int, default=42, help="Random seed for walk-forward sampling")
    p.add_argument("--wf-target-max-dd", type=float, default=35.0, help="Target max drawdown cap in percent for scoring")
    p.add_argument("--wf-min-trades", type=int, default=60, help="Minimum OOS trades required for candidate")
    p.add_argument("--max-open-positions", type=int, default=MAX_OPEN_POSITIONS, help=f"Maximum concurrent open positions (default: {MAX_OPEN_POSITIONS})")
    p.add_argument("--allocation-per-trade-pct", type=float, default=ALLOCATION_PER_TRADE_PCT, help=f"Capital allocation per trade in %% equity (default: {ALLOCATION_PER_TRADE_PCT})")
    p.add_argument("--initial-equity", type=float, default=INITIAL_EQUITY, help=f"Initial equity for portfolio simulation (default: {INITIAL_EQUITY})")

    return p.parse_args()


# Profile aggressive: ~905% return, DD ~73% (gần mục tiêu 1000%)
PROFILE_AGGRESSIVE = {
    "volume_spike_min_ratio": 0.5,
    "acc_range_max_pct": 60.0,
    "breakout_threshold_pct": 90.0,
    "atr_sl_multiplier": 0.35,
    "max_risk_per_trade_pct": 30.0,
    "rsi_min_daily": 26.0,
    "tp1_rr_ratio": 1.3,
    "tp2_rr_ratio": 15.0,
    "timeout_days": 90,
    "allocation_per_trade_pct": 25.0,
    "max_open_positions": 15,
}


def apply_overrides_from_args(args: argparse.Namespace) -> None:
    """Override tham số chiến lược từ CLI để tiện tối ưu/backtest nhanh."""
    global VOLUME_SPIKE_MIN_RATIO
    global ACCUMULATION_RANGE_MAX_PCT
    global BREAKOUT_THRESHOLD_PCT
    global ATR_SL_MULTIPLIER
    global MAX_RISK_PER_TRADE_PCT
    global RSI_MIN_DAILY
    global TP1_RR_RATIO
    global TP2_RR_RATIO
    global TAKER_FEE_BPS
    global SLIPPAGE_BPS
    global SLIPPAGE_ATR_MULT
    global MAX_TOTAL_SLIPPAGE_BPS
    global MAX_OPEN_POSITIONS
    global ALLOCATION_PER_TRADE_PCT
    global INITIAL_EQUITY

    if args.volume_spike_min_ratio is not None:
        VOLUME_SPIKE_MIN_RATIO = float(args.volume_spike_min_ratio)
    if args.acc_range_max_pct is not None:
        ACCUMULATION_RANGE_MAX_PCT = float(args.acc_range_max_pct)
    if args.breakout_threshold_pct is not None:
        BREAKOUT_THRESHOLD_PCT = float(args.breakout_threshold_pct)
    if args.atr_sl_multiplier is not None:
        ATR_SL_MULTIPLIER = float(args.atr_sl_multiplier)
    if args.max_risk_per_trade_pct is not None:
        MAX_RISK_PER_TRADE_PCT = float(args.max_risk_per_trade_pct)
    if args.rsi_min_daily is not None:
        RSI_MIN_DAILY = float(args.rsi_min_daily)
    if args.tp1_rr_ratio is not None:
        TP1_RR_RATIO = float(args.tp1_rr_ratio)
    if args.tp2_rr_ratio is not None:
        TP2_RR_RATIO = float(args.tp2_rr_ratio)
    TAKER_FEE_BPS = float(args.taker_fee_bps)
    SLIPPAGE_BPS = float(args.slippage_bps)
    SLIPPAGE_ATR_MULT = float(args.slippage_atr_mult)
    MAX_TOTAL_SLIPPAGE_BPS = float(args.max_total_slippage_bps)
    MAX_OPEN_POSITIONS = int(args.max_open_positions)
    ALLOCATION_PER_TRADE_PCT = float(args.allocation_per_trade_pct)
    INITIAL_EQUITY = float(args.initial_equity)

    # Profile aggressive overwrites (chạy cuối để đảm bảo áp dụng đúng)
    if args.profile == "aggressive":
        VOLUME_SPIKE_MIN_RATIO = float(PROFILE_AGGRESSIVE["volume_spike_min_ratio"])
        ACCUMULATION_RANGE_MAX_PCT = float(PROFILE_AGGRESSIVE["acc_range_max_pct"])
        BREAKOUT_THRESHOLD_PCT = float(PROFILE_AGGRESSIVE["breakout_threshold_pct"])
        ATR_SL_MULTIPLIER = float(PROFILE_AGGRESSIVE["atr_sl_multiplier"])
        MAX_RISK_PER_TRADE_PCT = float(PROFILE_AGGRESSIVE["max_risk_per_trade_pct"])
        RSI_MIN_DAILY = float(PROFILE_AGGRESSIVE["rsi_min_daily"])
        TP1_RR_RATIO = float(PROFILE_AGGRESSIVE["tp1_rr_ratio"])
        TP2_RR_RATIO = float(PROFILE_AGGRESSIVE["tp2_rr_ratio"])
        ALLOCATION_PER_TRADE_PCT = float(PROFILE_AGGRESSIVE["allocation_per_trade_pct"])
        MAX_OPEN_POSITIONS = int(PROFILE_AGGRESSIVE["max_open_positions"])
        args.allocation_per_trade_pct = ALLOCATION_PER_TRADE_PCT
        args.max_open_positions = MAX_OPEN_POSITIONS
        args.timeout_days = int(PROFILE_AGGRESSIVE["timeout_days"])


def _snapshot_strategy_params() -> dict[str, float]:
    return {
        "volume_spike_min_ratio": VOLUME_SPIKE_MIN_RATIO,
        "acc_range_max_pct": ACCUMULATION_RANGE_MAX_PCT,
        "breakout_threshold_pct": BREAKOUT_THRESHOLD_PCT,
        "atr_sl_multiplier": ATR_SL_MULTIPLIER,
        "max_risk_per_trade_pct": MAX_RISK_PER_TRADE_PCT,
        "rsi_min_daily": RSI_MIN_DAILY,
        "tp1_rr_ratio": TP1_RR_RATIO,
        "tp2_rr_ratio": TP2_RR_RATIO,
    }


def _apply_strategy_params(params: dict[str, float]) -> None:
    global VOLUME_SPIKE_MIN_RATIO
    global ACCUMULATION_RANGE_MAX_PCT
    global BREAKOUT_THRESHOLD_PCT
    global ATR_SL_MULTIPLIER
    global MAX_RISK_PER_TRADE_PCT
    global RSI_MIN_DAILY
    global TP1_RR_RATIO
    global TP2_RR_RATIO

    VOLUME_SPIKE_MIN_RATIO = float(params["volume_spike_min_ratio"])
    ACCUMULATION_RANGE_MAX_PCT = float(params["acc_range_max_pct"])
    BREAKOUT_THRESHOLD_PCT = float(params["breakout_threshold_pct"])
    ATR_SL_MULTIPLIER = float(params["atr_sl_multiplier"])
    MAX_RISK_PER_TRADE_PCT = float(params["max_risk_per_trade_pct"])
    RSI_MIN_DAILY = float(params["rsi_min_daily"])
    TP1_RR_RATIO = float(params["tp1_rr_ratio"])
    TP2_RR_RATIO = float(params["tp2_rr_ratio"])


def _build_walk_forward_windows(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_months: int,
    test_months: int,
    step_months: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cursor = pd.Timestamp(start_date)
    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end_date:
            break
        windows.append((cursor, train_end, test_end))
        cursor = cursor + pd.DateOffset(months=step_months)
    return windows


def _sample_walk_forward_candidates(max_combos: int, seed: int) -> list[dict[str, float]]:
    rnd = random.Random(seed)
    sampled: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()
    tries = 0

    while len(sampled) < max_combos and tries < max_combos * 40:
        tries += 1
        params = {
            "volume_spike_min_ratio": rnd.choice([0.6, 0.8, 1.0, 1.2, 1.5, 2.0]),
            "acc_range_max_pct": rnd.choice([25.0, 35.0, 50.0, 70.0]),
            "breakout_threshold_pct": rnd.choice([85.0, 90.0, 93.0, 95.0, 98.0]),
            "atr_sl_multiplier": rnd.choice([0.8, 1.0, 1.5, 2.0, 2.5]),
            "max_risk_per_trade_pct": rnd.choice([8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]),
            "rsi_min_daily": rnd.choice([35.0, 40.0, 45.0]),
            "tp1_rr_ratio": rnd.choice([1.0, 1.5, 2.0]),
            "tp2_rr_ratio": rnd.choice([2.0, 3.0, 4.0, 6.0, 8.0]),
            "timeout_days": rnd.choice([10.0, 20.0, 30.0, 45.0, 60.0]),
        }
        if params["tp2_rr_ratio"] <= params["tp1_rr_ratio"]:
            params["tp2_rr_ratio"] = params["tp1_rr_ratio"] + 1.0

        key = tuple(float(params[k]) for k in sorted(params.keys()))
        if key in seen:
            continue
        seen.add(key)
        sampled.append(params)

    return sampled


def _prepare_symbol_cache(data_dir: Path, symbols: list[str]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for sym in symbols:
        df_1h = load_ohlcv(data_dir, sym, "1h")
        if df_1h is None or len(df_1h) < VOLUME_SMA_PERIOD + 10:
            continue
        df_d = load_ohlcv(data_dir, sym, "1d")
        if df_d is None:
            df_d = resample_to_daily(df_1h)
        if len(df_d) < WARMUP_DAILY_BARS:
            continue
        cache[sym] = (df_1h, df_d)
    return cache


def _run_portfolio_on_window(
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    enable_btc_filter: bool,
    timeout_days: int,
    taker_fee_bps: float,
    slippage_bps: float,
    slippage_atr_mult: float,
    max_total_slippage_bps: float,
    allocation_per_trade_pct: float | None = None,
    max_open_positions: int | None = None,
    initial_equity: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    btc_d_indicators = None
    local_btc_filter = enable_btc_filter
    if local_btc_filter and BTC_SYMBOL in cache:
        _, btc_d_all = cache[BTC_SYMBOL]
        btc_d = btc_d_all[start_date - pd.Timedelta(days=WARMUP_DAILY_BARS + 5): end_date]
        if len(btc_d) >= WARMUP_DAILY_BARS:
            btc_d_indicators = compute_daily_indicators(btc_d)
        else:
            local_btc_filter = False
    elif local_btc_filter:
        local_btc_filter = False

    all_trades_list: list[pd.DataFrame] = []
    for sym, (df_1h_all, df_d_all) in cache.items():
        df_1h = df_1h_all[start_date: end_date]
        df_d = df_d_all[start_date - pd.Timedelta(days=WARMUP_DAILY_BARS + 5): end_date]
        if df_1h.empty or len(df_d) < WARMUP_DAILY_BARS:
            continue

        df_d_ind = compute_daily_indicators(df_d)
        merged = merge_daily_onto_1h(df_1h, df_d_ind, btc_d_indicators, local_btc_filter and sym != BTC_SYMBOL)
        signals = extract_signals(merged, sym)
        if signals.empty:
            continue
        trades = simulate_trades(
            df_1h,
            signals,
            timeout_days=timeout_days,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
            slippage_atr_mult=slippage_atr_mult,
            max_total_slippage_bps=max_total_slippage_bps,
        )
        if not trades.empty:
            all_trades_list.append(trades)

    if not all_trades_list:
        return pd.DataFrame(), {"total_trades": 0}

    raw_trades = pd.concat(all_trades_list, ignore_index=True).sort_values("entry_time")
    alloc_pct = allocation_per_trade_pct if allocation_per_trade_pct is not None else ALLOCATION_PER_TRADE_PCT
    max_pos = max_open_positions if max_open_positions is not None else MAX_OPEN_POSITIONS
    eq0 = initial_equity if initial_equity is not None else INITIAL_EQUITY
    executed_trades, portfolio_metrics, _ = apply_portfolio_allocation(
        raw_trades,
        initial_equity=eq0,
        allocation_per_trade_pct=alloc_pct,
        max_open_positions=max_pos,
    )
    return executed_trades, portfolio_metrics


def _wf_score(metrics: dict, target_max_dd: float, min_trades: int) -> float:
    if metrics.get("total_trades", 0) < min_trades:
        return -1e9 + metrics.get("total_trades", 0)
    ret = float(metrics.get("total_return_pct", -1e9))
    dd = abs(float(metrics.get("max_drawdown_pct", 0.0)))
    pf = float(metrics.get("profit_factor", 0.0))
    if dd > target_max_dd:
        # Hard constraint: reject cấu hình vượt trần drawdown mục tiêu.
        return -1e8 - dd
    # Trong vùng DD cho phép: tối ưu return, thưởng nhẹ PF.
    return ret + min(pf, 5.0) * 4.0


def run_walk_forward(args: argparse.Namespace, symbols: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp, enable_btc_filter: bool) -> None:
    logger.info("WALK-FORWARD mode ON")
    windows = _build_walk_forward_windows(
        start_date=start_date,
        end_date=end_date,
        train_months=args.wf_train_months,
        test_months=args.wf_test_months,
        step_months=args.wf_step_months,
    )
    if not windows:
        logger.error("No walk-forward windows created. Check start/end and wf window sizes.")
        return

    cache = _prepare_symbol_cache(args.data_dir, symbols)
    if not cache:
        logger.error("No valid symbols after data filtering for walk-forward.")
        return

    candidates = _sample_walk_forward_candidates(args.wf_max_combos, args.wf_seed)
    logger.info("WF windows: %d | candidates: %d | cached symbols: %d", len(windows), len(candidates), len(cache))

    original_params = _snapshot_strategy_params()
    best_candidate: dict | None = None
    best_oos_trades = pd.DataFrame()
    best_window_rows: list[dict] = []

    try:
        for idx, cand in enumerate(candidates, 1):
            _apply_strategy_params(cand)
            timeout_days = int(cand["timeout_days"])
            oos_trades_list: list[pd.DataFrame] = []
            window_rows: list[dict] = []

            for w_idx, (train_start, train_end, test_end) in enumerate(windows, 1):
                _, train_m = _run_portfolio_on_window(
                    cache, train_start, train_end, enable_btc_filter, timeout_days,
                    args.taker_fee_bps, args.slippage_bps, args.slippage_atr_mult, args.max_total_slippage_bps,
                )
                test_trades, test_m = _run_portfolio_on_window(
                    cache, train_end, test_end, enable_btc_filter, timeout_days,
                    args.taker_fee_bps, args.slippage_bps, args.slippage_atr_mult, args.max_total_slippage_bps,
                )
                if not test_trades.empty:
                    test_trades = test_trades.copy()
                    test_trades["wf_window"] = w_idx
                    oos_trades_list.append(test_trades)

                window_rows.append({
                    "window": w_idx,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": train_end,
                    "test_end": test_end,
                    "train_return_pct": train_m.get("total_return_pct", 0.0),
                    "train_max_dd_pct": train_m.get("max_drawdown_pct", 0.0),
                    "test_return_pct": test_m.get("total_return_pct", 0.0),
                    "test_max_dd_pct": test_m.get("max_drawdown_pct", 0.0),
                    "test_trades": test_m.get("total_trades", 0),
                })

            if not oos_trades_list:
                continue
            oos_trades = pd.concat(oos_trades_list, ignore_index=True).sort_values("entry_time")
            oos_metrics = compute_metrics(oos_trades)
            score = _wf_score(oos_metrics, args.wf_target_max_dd, args.wf_min_trades)

            if best_candidate is None or score > best_candidate["score"]:
                best_candidate = {
                    "score": score,
                    "params": dict(cand),
                    "oos_metrics": dict(oos_metrics),
                    "timeout_days": timeout_days,
                }
                best_oos_trades = oos_trades.copy()
                best_window_rows = window_rows
                logger.info(
                    "NEW WF BEST %d/%d | score=%.2f return=%+.2f%% maxdd=%.2f%% trades=%d | params=%s",
                    idx, len(candidates), score, oos_metrics.get("total_return_pct", 0.0),
                    oos_metrics.get("max_drawdown_pct", 0.0), oos_metrics.get("total_trades", 0),
                    cand,
                )
    finally:
        _apply_strategy_params(original_params)

    if best_candidate is None:
        logger.warning("WF finished: no candidate satisfied evaluation.")
        return

    wf_dir = args.output_dir / "walk_forward"
    wf_dir.mkdir(parents=True, exist_ok=True)
    best_oos_trades.to_csv(wf_dir / "wf_oos_trades.csv", index=False)
    pd.DataFrame(best_window_rows).to_csv(wf_dir / "wf_windows.csv", index=False)

    summary = {
        "best_score": best_candidate["score"],
        "best_params_json": json.dumps(best_candidate["params"], ensure_ascii=True),
        **best_candidate["oos_metrics"],
        "taker_fee_bps": args.taker_fee_bps,
        "slippage_bps": args.slippage_bps,
        "slippage_atr_mult": args.slippage_atr_mult,
        "max_total_slippage_bps": args.max_total_slippage_bps,
        "windows": len(windows),
        "candidates": len(candidates),
    }
    pd.DataFrame([summary]).to_csv(wf_dir / "wf_summary.csv", index=False)

    logger.info("=" * 70)
    logger.info("WALK-FORWARD BEST CONFIG")
    logger.info("  Score:          %.2f", best_candidate["score"])
    logger.info("  Total Return:   %+.2f%%", best_candidate["oos_metrics"]["total_return_pct"])
    logger.info("  Max Drawdown:   %.2f%%", best_candidate["oos_metrics"]["max_drawdown_pct"])
    logger.info("  Win Rate:       %.1f%%", best_candidate["oos_metrics"]["win_rate_pct"])
    logger.info("  Profit Factor:  %.2f", best_candidate["oos_metrics"]["profit_factor"])
    logger.info("  Trades:         %d", best_candidate["oos_metrics"]["total_trades"])
    logger.info("  Params:         %s", best_candidate["params"])
    logger.info("WF outputs: %s", wf_dir)
    logger.info("=" * 70)


def main() -> None:
    args = parse_args()
    apply_overrides_from_args(args)

    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir
    enable_btc_filter: bool = BACKTEST_ENABLE_BTC_FILTER_DEFAULT
    if args.btc_filter:
        enable_btc_filter = True
    if args.no_btc_filter:
        enable_btc_filter = False

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
    logger.info(
        "COSTS     │  fee=%.2fbps  base_slippage=%.2fbps  atr_mult=%.3f  cap=%.2fbps",
        args.taker_fee_bps,
        args.slippage_bps,
        args.slippage_atr_mult,
        args.max_total_slippage_bps,
    )
    logger.info("=" * 70)

    if args.walk_forward:
        run_walk_forward(args, symbols, start_date, end_date, enable_btc_filter)
        return

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
                sym, data_dir, btc_d_indicators, False, start_date, end_date, args.timeout_days,
                args.taker_fee_bps, args.slippage_bps, args.slippage_atr_mult, args.max_total_slippage_bps
            )
        else:
            trades, metrics = run_backtest_for_symbol(
                sym, data_dir, btc_d_indicators, enable_btc_filter, start_date, end_date, args.timeout_days,
                args.taker_fee_bps, args.slippage_bps, args.slippage_atr_mult, args.max_total_slippage_bps
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

    raw_all_trades = pd.concat(all_trades_list, ignore_index=True).sort_values("entry_time")
    raw_all_trades_path = output_dir / "all_trades_raw.csv"
    raw_all_trades.to_csv(raw_all_trades_path, index=False)

    # Portfolio metrics theo kiểu thực chiến: giới hạn concurrent positions + allocation theo equity
    all_trades, portfolio_metrics, equity_df = apply_portfolio_allocation(
        raw_all_trades,
        initial_equity=args.initial_equity,
        allocation_per_trade_pct=args.allocation_per_trade_pct,
        max_open_positions=args.max_open_positions,
    )
    all_trades_path = output_dir / "all_trades.csv"
    all_trades.to_csv(all_trades_path, index=False)
    logger.info(
        "Saved %d executed trades (from %d raw signals) → %s",
        len(all_trades),
        len(raw_all_trades),
        all_trades_path,
    )

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
    logger.info("  Initial Equity: %.2f  |  Final Equity: %.2f",
                portfolio_metrics.get("initial_equity", args.initial_equity),
                portfolio_metrics.get("final_equity", args.initial_equity))
    logger.info("  Position Model: max_open=%d  alloc_per_trade=%.1f%%  skipped(capacity/cash)=%d/%d",
                args.max_open_positions,
                args.allocation_per_trade_pct,
                portfolio_metrics.get("skipped_capacity", 0),
                portfolio_metrics.get("skipped_cash", 0))
    logger.info("=" * 70)

    # Equity curve (thực chiến)
    eq_path = output_dir / "equity_curve.csv"
    equity_df.to_csv(eq_path, index=False)

    # Portfolio summary CSV
    summary_rows = [{"symbol": sym, **m} for sym, m in all_metrics.items() if m.get("total_trades", 0) > 0]
    summary_rows.append({"symbol": "_PORTFOLIO_", **portfolio_metrics})
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)

    logger.info("")
    logger.info("Output files:")
    logger.info("  %s  (all trades)", all_trades_path)
    logger.info("  %s  (all raw trades before allocation)", raw_all_trades_path)
    logger.info("  %s  (equity curve)", eq_path)
    logger.info("  %s  (summary)", output_dir / "summary.csv")
    logger.info("  %s/*.csv  (per-symbol trades)", output_dir)


if __name__ == "__main__":
    main()
