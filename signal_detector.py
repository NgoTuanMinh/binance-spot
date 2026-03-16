"""
Logic phân tích tín hiệu BUY theo bộ lọc 3 lớp (swing trading).
Dùng dữ liệu OHLCV từ Binance qua CCXT.
"""
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ACCUMULATION_BOTTOM_PCT,
    ACCUMULATION_LOOKBACK,
    ACCUMULATION_RANGE_MAX_PCT,
    ATR_PERIOD,
    BREAKOUT_THRESHOLD_PCT,
    CANDLES_1D,
    CANDLES_1H,
    CANDLES_4H,
    EMA_PERIODS,
    MIN_VOLUME_USDT,
    RSI_MIN_DAILY,
    RSI_PERIOD,
    SUPPORT_RESISTANCE_PERIOD,
    VOLUME_SMA_PERIOD,
    VOLUME_SPIKE_MIN_RATIO,
)

logger = logging.getLogger(__name__)


@dataclass
class BuySignal:
    """Kết quả khi phát hiện tín hiệu BUY."""
    symbol: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    risk_pct: float
    reward_tp1_pct: float
    rr_ratio: float
    volume_ratio: float
    accumulation_range_pct: float
    rsi_daily: float
    atr_pct_1h: float
    raw: dict[str, Any]  # dữ liệu thô để log/debug


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _ensure_dataframe(ohlcv: list) -> pd.DataFrame:
    """Chuyển list OHLCV từ CCXT thành DataFrame chuẩn."""
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(how="all")


class SignalDetector:
    """
    Phát hiện tín hiệu BUY theo 3 lớp:
    - Lớp 1: Xu hướng dài hạn (Daily) - Giá > EMA50, EMA50 > EMA200, RSI > 40
    - Lớp 2: Volume spike - Volume hiện tại > 600% SMA(volume, 65) trên khung 1h
    - Lớp 3: Breakout từ vùng tích lũy (range 30 nến < 15%, giá gần đáy, breakout >= 98% resistance)
    """

    def __init__(self, exchange: Any):
        self.exchange = exchange

    def _fetch_ohlcv_with_retry(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        max_retries: int = 3,
    ) -> list:
        """Lấy OHLCV với exponential backoff, tránh rate limit."""
        for attempt in range(max_retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if not ohlcv or len(ohlcv) < limit:
                    logger.warning(
                        "fetch_ohlcv %s %s got %d candles (expected %d)",
                        symbol, timeframe, len(ohlcv) if ohlcv else 0, limit,
                    )
                return ohlcv or []
            except Exception as e:
                wait = (2 ** attempt) + 1
                logger.warning("fetch_ohlcv attempt %d failed: %s, retry in %ds", attempt + 1, e, wait)
                if attempt == max_retries - 1:
                    raise
                time.sleep(wait)
        return []

    def _get_daily_indicators(self, symbol: str) -> pd.DataFrame | None:
        """Lấy 1d và tính EMA(20,50,200), RSI(14) trên daily."""
        ohlcv = self._fetch_ohlcv_with_retry(symbol, "1d", CANDLES_1D)
        if len(ohlcv) < 200:
            return None
        df = _ensure_dataframe(ohlcv)
        close = df["close"]
        for period in EMA_PERIODS:
            df[f"ema_{period}"] = _ema(close, period)
        df["rsi"] = _rsi(close, RSI_PERIOD)
        return df

    def _get_4h_1h_volume_atr(self, symbol: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """
        Lấy dữ liệu 1h (volume SMA 65 + ATR) và 4h (hiện tại không bắt buộc, giữ để dễ mở rộng).
        Volume spike được tính trên khung 1h.
        """
        ohlcv_4h = self._fetch_ohlcv_with_retry(symbol, "4h", CANDLES_4H)
        ohlcv_1h = self._fetch_ohlcv_with_retry(symbol, "1h", CANDLES_1H)
        # Volume spike và ATR đều dùng 1h nên cần đủ nến 1h
        if len(ohlcv_1h) < max(VOLUME_SMA_PERIOD, ATR_PERIOD + 5):
            return None, None
        df_4h = _ensure_dataframe(ohlcv_4h) if ohlcv_4h else pd.DataFrame()
        df_1h = _ensure_dataframe(ohlcv_1h)
        # Volume SMA 65 trên khung 1h
        df_1h["volume_sma"] = _sma(df_1h["volume"], VOLUME_SMA_PERIOD)
        df_1h["atr"] = _atr(df_1h["high"], df_1h["low"], df_1h["close"], ATR_PERIOD)
        return df_4h, df_1h

    def _layer1_trend(self, df_d: pd.DataFrame) -> bool:
        """Lớp 1: Giá > EMA50, EMA50 > EMA200, RSI(14) > 40 (daily)."""
        if df_d.empty or len(df_d) < 2:
            return False
        last = df_d.iloc[-1]
        price = last["close"]
        ema50 = last["ema_50"]
        ema200 = last["ema_200"]
        rsi = last["rsi"]
        if pd.isna(ema50) or pd.isna(ema200) or pd.isna(rsi):
            return False
        return price > ema50 and ema50 > ema200 and rsi > RSI_MIN_DAILY

    def _layer2_volume_spike(self, df_1h: pd.DataFrame) -> tuple[bool, float]:
        """Lớp 2: volume_ratio = current_volume_1h / SMA(volume_1h, 65) > 2.5. Trả về (pass, ratio)."""
        if df_1h.empty or len(df_1h) < VOLUME_SMA_PERIOD:
            return False, 0.0
        last = df_1h.iloc[-1]
        vol = last["volume"]
        vol_sma = last["volume_sma"]
        if pd.isna(vol_sma) or vol_sma <= 0:
            return False, 0.0
        ratio = float(vol / vol_sma)
        return ratio >= VOLUME_SPIKE_MIN_RATIO, ratio

    def _layer3_accumulation_breakout(
        self,
        df_d: pd.DataFrame,
        current_price: float,
    ) -> tuple[bool, float]:
        """
        Lớp 3: Vùng tích lũy 30 nến (range < 15%), giá gần đáy (30% dưới),
        breakout khi giá >= 98% kháng cự. Trả về (pass, accumulation_range_pct).
        """
        if df_d.empty or len(df_d) < ACCUMULATION_LOOKBACK:
            return False, 0.0
        window = df_d.iloc[-ACCUMULATION_LOOKBACK:]
        low = window["low"].min()
        high = window["high"].max()
        range_pct = ((high - low) / low * 100) if low > 0 else 999.0
        if range_pct >= ACCUMULATION_RANGE_MAX_PCT:
            return False, range_pct
        # Giá gần đáy: trong 30% dưới của range (đóng cửa nến trước nằm trong vùng đáy)
        bottom_level = low + (high - low) * (ACCUMULATION_BOTTOM_PCT / 100)
        prev_close = df_d.iloc[-2]["close"] if len(df_d) >= 2 else df_d.iloc[-1]["close"]
        if prev_close > bottom_level:
            return False, range_pct
        # Breakout: giá hiện tại vượt 98% kháng cự vùng tích lũy
        resistance = high
        breakout_level = resistance * (BREAKOUT_THRESHOLD_PCT / 100)
        if current_price < breakout_level:
            return False, range_pct
        return True, range_pct

    def _support_resistance_20(self, df: pd.DataFrame) -> tuple[float, float]:
        """Kháng cự/ hỗ trợ động 20 period. Trả về (support, resistance)."""
        if df.empty or len(df) < SUPPORT_RESISTANCE_PERIOD:
            return 0.0, 0.0
        window = df.iloc[-SUPPORT_RESISTANCE_PERIOD:]
        return float(window["low"].min()), float(window["high"].max())

    def _compute_sl_tp(
        self,
        entry: float,
        atr_1h: float,
        accumulation_low: float,
    ) -> tuple[float, float, float]:
        """
        Stop loss: dưới entry, có thể dùng max(accumulation_low, entry - 2*ATR).
        TP1 = +10%, TP2 = +20%.
        """
        # SL: dưới đáy tích lũy hoặc entry - 1.5 ATR (tránh quá chặt)
        sl_candidate_atr = entry - 1.5 * atr_1h
        sl = min(accumulation_low, sl_candidate_atr) if accumulation_low > 0 else sl_candidate_atr
        if sl >= entry:
            sl = entry - atr_1h
        tp1 = entry * 1.10
        tp2 = entry * 1.20
        return sl, tp1, tp2

    def check_buy_signal(self, symbol: str) -> BuySignal | None:
        """
        Kiểm tra đủ 3 lớp lọc; nếu pass thì tính entry, SL, TP và trả về BuySignal.
        """
        try:
            df_d = self._get_daily_indicators(symbol)
            if df_d is None:
                logger.debug("%s: insufficient daily data", symbol)
                return None
            if not self._layer1_trend(df_d):
                logger.debug("%s: layer1 trend failed", symbol)
                return None

            df_4h, df_1h = self._get_4h_1h_volume_atr(symbol)
            if df_1h is None or df_1h.empty:
                return None
            pass_vol, volume_ratio = self._layer2_volume_spike(df_1h)
            if not pass_vol:
                logger.debug("%s: layer2 volume spike failed (ratio=%.2f)", symbol, volume_ratio)
                return None

            current_price = float(df_d.iloc[-1]["close"])
            pass_acc, acc_range_pct = self._layer3_accumulation_breakout(df_d, current_price)
            if not pass_acc:
                logger.debug("%s: layer3 accumulation/breakout failed", symbol)
                return None

            # Lọc volume USDT tối thiểu (optional: cần 24h volume từ exchange)
            # Ở đây dùng volume 1h gần nhất * 24 làm proxy 24h (1h * 24 = 24h)
            last_vol = float(df_1h.iloc[-1]["volume"])
            proxy_24h_quote_vol = last_vol * current_price * 24
            if proxy_24h_quote_vol < MIN_VOLUME_USDT:
                logger.debug("%s: min volume USDT not met", symbol)
                return None

            # Accumulation low cho SL
            window = df_d.iloc[-ACCUMULATION_LOOKBACK:]
            acc_low = float(window["low"].min())
            acc_high = float(window["high"].max())

            # ATR 1h (điểm) và ATR% (cho message)
            if df_1h is not None and not df_1h.empty:
                atr_val = float(df_1h.iloc[-1]["atr"])
                atr_pct = (atr_val / current_price * 100) if current_price > 0 else 0.0
            else:
                atr_val = (acc_high - acc_low) * 0.02
                atr_pct = 0.0

            entry = current_price
            sl, tp1, tp2 = self._compute_sl_tp(entry, atr_val, acc_low)
            risk_pct = ((entry - sl) / entry * 100) if entry > 0 else 0.0
            reward_tp1_pct = 10.0
            rr = (tp1 - entry) / (entry - sl) if (entry - sl) > 0 else 0.0

            rsi_daily = float(df_d.iloc[-1]["rsi"])

            return BuySignal(
                symbol=symbol,
                entry=entry,
                stop_loss=sl,
                tp1=tp1,
                tp2=tp2,
                risk_pct=risk_pct,
                reward_tp1_pct=reward_tp1_pct,
                rr_ratio=rr,
                volume_ratio=volume_ratio,
                accumulation_range_pct=acc_range_pct,
                rsi_daily=rsi_daily,
                atr_pct_1h=atr_pct,
                raw={
                    "acc_low": acc_low,
                    "acc_high": acc_high,
                },
            )
        except Exception as e:
            logger.exception("check_buy_signal %s error: %s", symbol, e)
            return None
