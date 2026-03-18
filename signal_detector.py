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
    ACCUMULATION_POSITION_PCT,
    ACCUMULATION_LOOKBACK,
    ACCUMULATION_RANGE_MAX_PCT,
    ATR_PERIOD,
    ATR_SL_MULTIPLIER,
    BREAKOUT_THRESHOLD_PCT,
    CANDLES_1D,
    CANDLES_1H,
    EMA_PERIODS,
    ENABLE_BTC_FILTER,
    MIN_VOLUME_USDT,
    RSI_MIN_DAILY,
    RSI_PERIOD,
    VOLUME_SMA_PERIOD,
    VOLUME_SPIKE_LOOKBACK_CANDLES,
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
    - Lớp 0 (tiền đề): BTC phải uptrend trên Daily (tùy chọn, xem ENABLE_BTC_FILTER)
    - Lớp 1: Xu hướng dài hạn (Daily) - Giá > EMA50, EMA50 > EMA200, RSI > 40
    - Lớp 2: Volume spike - Volume của nến 1h VỪA đóng > VOLUME_SPIKE_MIN_RATIO × SMA(volume, 65)
    - Lớp 3: Breakout từ vùng tích lũy (range 30 nến đã đóng < 15%, nến trước ở đáy,
              giá hiện tại >= 98% kháng cự)
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

    def _get_1h_volume_atr(self, symbol: str) -> pd.DataFrame | None:
        """
        Lấy dữ liệu 1h: tính Volume SMA(65) và ATR(14).
        Trả về None nếu không đủ dữ liệu.
        """
        ohlcv_1h = self._fetch_ohlcv_with_retry(symbol, "1h", CANDLES_1H)
        # Cần đủ nến để SMA warmup + ít nhất 2 nến đã đóng để dùng iloc[-2]
        if len(ohlcv_1h) < max(VOLUME_SMA_PERIOD + 2, ATR_PERIOD + 5):
            return None
        df_1h = _ensure_dataframe(ohlcv_1h)
        df_1h["volume_sma"] = _sma(df_1h["volume"], VOLUME_SMA_PERIOD)
        df_1h["atr"] = _atr(df_1h["high"], df_1h["low"], df_1h["close"], ATR_PERIOD)
        return df_1h

    # ------------------------------------------------------------------
    # Lớp 0: Bộ lọc thị trường chung (BTC)
    # ------------------------------------------------------------------

    def check_market_trend(self, btc_symbol: str) -> bool:
        """
        Kiểm tra xu hướng thị trường chung qua BTC:
        - Giá BTC > EMA50 (daily)
        - EMA50 > EMA200 (daily)
        Nếu ENABLE_BTC_FILTER=False thì luôn trả True.
        """
        if not ENABLE_BTC_FILTER:
            return True
        try:
            df = self._get_daily_indicators(btc_symbol)
            if df is None or df.empty:
                logger.warning("BTC daily data unavailable, skipping BTC filter")
                return True
            # Dùng nến đã đóng gần nhất (iloc[-2]) để tránh candle đang hình thành
            last = df.iloc[-2]
            ema50 = last["ema_50"]
            ema200 = last["ema_200"]
            price = last["close"]
            if pd.isna(ema50) or pd.isna(ema200):
                return True
            trend_ok = price > ema50 and ema50 > ema200
            if not trend_ok:
                logger.info("BTC market filter: DOWNTREND (price=%.2f ema50=%.2f ema200=%.2f) → skip all", price, ema50, ema200)
            return trend_ok
        except Exception as e:
            logger.warning("check_market_trend BTC error: %s — skipping filter", e)
            return True

    # ------------------------------------------------------------------
    # Lớp 1: Xu hướng dài hạn (Daily)
    # ------------------------------------------------------------------

    def _layer1_trend(self, df_d: pd.DataFrame) -> bool:
        """
        Lớp 1: Dùng nến daily đã đóng gần nhất (iloc[-2]).
        Điều kiện: Giá > EMA50, EMA50 > EMA200, RSI(14) > RSI_MIN_DAILY.
        """
        if df_d.empty or len(df_d) < 3:
            return False
        # iloc[-2]: nến daily đã đóng hoàn toàn (iloc[-1] có thể chưa xong)
        last = df_d.iloc[-2]
        price = last["close"]
        ema50 = last["ema_50"]
        ema200 = last["ema_200"]
        rsi = last["rsi"]
        if pd.isna(ema50) or pd.isna(ema200) or pd.isna(rsi):
            return False
        return price > ema50 and ema50 > ema200 and rsi > RSI_MIN_DAILY

    # ------------------------------------------------------------------
    # Lớp 2: Volume spike trên 1h
    # ------------------------------------------------------------------

    def _layer2_volume_spike(self, df_1h: pd.DataFrame) -> tuple[bool, float]:
        """
        Lớp 2: Xét VOLUME_SPIKE_LOOKBACK_CANDLES nến 1H đã đóng gần nhất (mặc định 2 nến).
        Pass nếu BẤT KỲ nến nào trong cửa sổ đó có ratio >= VOLUME_SPIKE_MIN_RATIO (6x).
        Trả về (pass, ratio_cao_nhất_trong_cửa_sổ).

        Lý do xét nhiều nến: bot scan mỗi 1H nên có thể bị trễ nhẹ vài phút;
        xét 2 nến đã đóng đảm bảo không bỏ sót spike ở nến ngay trước đó.
        """
        min_required = VOLUME_SMA_PERIOD + VOLUME_SPIKE_LOOKBACK_CANDLES + 1
        if df_1h.empty or len(df_1h) < min_required:
            return False, 0.0

        # Lấy N nến đã đóng: bỏ iloc[-1] (đang hình thành), lấy ngược từ iloc[-2]
        window = df_1h.iloc[-(VOLUME_SPIKE_LOOKBACK_CANDLES + 1): -1]
        best_ratio = 0.0
        for _, row in window.iterrows():
            vol = row["volume"]
            vol_sma = row["volume_sma"]
            if pd.isna(vol_sma) or vol_sma <= 0 or pd.isna(vol):
                continue
            ratio = float(vol / vol_sma)
            if ratio > best_ratio:
                best_ratio = ratio

        return best_ratio >= VOLUME_SPIKE_MIN_RATIO, best_ratio

    # ------------------------------------------------------------------
    # Lớp 3: Breakout từ vùng tích lũy (Daily)
    # ------------------------------------------------------------------

    def _layer3_accumulation_breakout(
        self,
        df_d: pd.DataFrame,
        current_price: float,
    ) -> tuple[bool, float]:
        """
        Lớp 3: Vùng tích lũy được xác định từ ACCUMULATION_LOOKBACK nến daily ĐÃ ĐÓNG
        (loại trừ candle đang hình thành).
        Điều kiện:
        1. Range tích lũy < ACCUMULATION_RANGE_MAX_PCT (15%)
        2. Nến daily đóng cửa gần nhất nằm trong 30% dưới của range
        3. Giá hiện tại >= BREAKOUT_THRESHOLD_PCT% của kháng cự (breakout)
        Trả về (pass, accumulation_range_pct).
        """
        # Cần đủ nến completed: LOOKBACK + 1 (hiện tại) + 1 (prev)
        if df_d.empty or len(df_d) < ACCUMULATION_LOOKBACK + 2:
            return False, 0.0

        # Dùng ACCUMULATION_LOOKBACK nến đã đóng, bỏ candle hiện tại (iloc[-1])
        window = df_d.iloc[-ACCUMULATION_LOOKBACK - 1: -1]
        low = window["low"].min()
        high = window["high"].max()
        range_pct = ((high - low) / low * 100) if low > 0 else 999.0
        if range_pct >= ACCUMULATION_RANGE_MAX_PCT:
            return False, range_pct

        # Nến daily đã đóng gần nhất phải nằm ở NỬA TRÊN (hoặc mức POSITION_PCT) của range
        # Mô hình VCP (Volatility Contraction) - tích lũy sát kháng cự
        position_level = low + (high - low) * (ACCUMULATION_POSITION_PCT / 100)
        prev_close = df_d.iloc[-2]["close"]
        if prev_close < position_level:
            return False, range_pct

        # Giá hiện tại (có thể là candle đang hình thành) phải vượt 98% kháng cự
        resistance = high
        breakout_level = resistance * (BREAKOUT_THRESHOLD_PCT / 100)
        if current_price < breakout_level:
            return False, range_pct

        return True, range_pct

    # ------------------------------------------------------------------
    # Tính SL / TP
    # ------------------------------------------------------------------

    def _compute_sl_tp(
        self,
        entry: float,
        atr_1h: float,
        accumulation_low: float,
    ) -> tuple[float, float, float]:
        """
        SL: min(accumulation_low, entry - ATR_SL_MULTIPLIER × ATR_1h).
        Dùng min để lấy mức SL RỘNG HƠN, ưu tiên đáy tích lũy.
        TP1 = entry × 1.10, TP2 = entry × 1.20.

        Lý do dùng min:
        - `accumulation_low` là hỗ trợ tự nhiên, SL dưới mức này là hợp lý.
        - `entry - N×ATR` có thể đặt SL quá gần nếu ATR nhỏ.
        - `min` đảm bảo SL không bị đặt quá chặt, cho giá có không gian "thở".
        """
        sl_atr = entry - ATR_SL_MULTIPLIER * atr_1h
        sl = min(accumulation_low, sl_atr) if accumulation_low > 0 else sl_atr
        if sl >= entry:
            sl = entry - atr_1h  # fallback tránh SL >= entry
        tp1 = entry * 1.10
        tp2 = entry * 1.20
        return sl, tp1, tp2

    # ------------------------------------------------------------------
    # Entry point chính
    # ------------------------------------------------------------------

    def check_buy_signal(self, symbol: str) -> BuySignal | None:
        """
        Kiểm tra đủ 3 lớp lọc; nếu pass thì tính entry, SL, TP và trả về BuySignal.
        BTC filter (lớp 0) được kiểm tra 1 lần/scan trong run_scan() trước khi gọi hàm này.
        """
        try:
            df_d = self._get_daily_indicators(symbol)
            if df_d is None:
                logger.debug("%s: insufficient daily data", symbol)
                return None
            if not self._layer1_trend(df_d):
                logger.debug("%s: layer1 trend failed", symbol)
                return None

            df_1h = self._get_1h_volume_atr(symbol)
            if df_1h is None or df_1h.empty:
                logger.debug("%s: insufficient 1h data", symbol)
                return None

            pass_vol, volume_ratio = self._layer2_volume_spike(df_1h)
            if not pass_vol:
                logger.debug("%s: layer2 volume spike failed (ratio=%.2f)", symbol, volume_ratio)
                return None

            # Dùng giá đóng cửa hiện tại (có thể là nến đang hình thành) làm current price
            current_price = float(df_d.iloc[-1]["close"])
            pass_acc, acc_range_pct = self._layer3_accumulation_breakout(df_d, current_price)
            if not pass_acc:
                logger.debug("%s: layer3 accumulation/breakout failed", symbol)
                return None

            # Lọc volume USDT tối thiểu: dùng nến 1h đã đóng × giá × 24 làm proxy 24h
            last_completed_vol = float(df_1h.iloc[-2]["volume"])
            proxy_24h_quote_vol = last_completed_vol * current_price * 24
            if proxy_24h_quote_vol < MIN_VOLUME_USDT:
                logger.debug("%s: min volume USDT not met (proxy=%.0f)", symbol, proxy_24h_quote_vol)
                return None

            # Accumulation zone từ nến đã đóng (giống Layer 3)
            window = df_d.iloc[-ACCUMULATION_LOOKBACK - 1: -1]
            acc_low = float(window["low"].min())
            acc_high = float(window["high"].max())

            atr_val = float(df_1h.iloc[-2]["atr"])
            atr_pct = (atr_val / current_price * 100) if current_price > 0 else 0.0

            entry = current_price
            sl, tp1, tp2 = self._compute_sl_tp(entry, atr_val, acc_low)

            risk_pct = ((entry - sl) / entry * 100) if entry > 0 else 0.0
            reward_tp1_pct = ((tp1 - entry) / entry * 100) if entry > 0 else 10.0
            rr = (tp1 - entry) / (entry - sl) if (entry - sl) > 0 else 0.0
            rsi_daily = float(df_d.iloc[-2]["rsi"])

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
