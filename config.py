"""
Cấu hình bot từ biến môi trường (.env).
"""
import os
from pathlib import Path

from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]

# Load .env từ thư mục gốc project
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


def _get_env(key: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(f"Missing required env: {key}")
    return value or ""


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


# --- Telegram ---
TELEGRAM_BOT_TOKEN = _get_env("TELEGRAM_BOT_TOKEN", required=True)
TELEGRAM_CHAT_ID = _get_env("TELEGRAM_CHAT_ID", required=True)

# --- Symbols (Binance spot) ---
# Nếu set SYMBOLS thì dùng list cố định; để trống thì lấy top TOP_SYMBOLS_COUNT theo volume 24h
SYMBOLS_STR = _get_env("SYMBOLS", "")
SYMBOLS = [s.strip() for s in SYMBOLS_STR.split(",") if s.strip()]
# Top 10 -> 100 symbol theo khối lượng 24h (chỉ dùng khi SYMBOLS trống)
TOP_SYMBOLS_COUNT = max(10, min(100, _get_int("TOP_SYMBOLS_COUNT", 50)))
USE_TOP_BY_VOLUME = len(SYMBOLS) == 0

# --- Scheduling ---
# Nên đặt = 1 vì Layer 2 dùng tín hiệu volume 1H; scan 4H sẽ bỏ sót 3/4 nến
POLLING_INTERVAL_HOURS = _get_int("POLLING_INTERVAL_HOURS", 1)

# Số nến 1H gần nhất được xét cho volume spike (để tránh bỏ sót khi scan bị delay nhẹ)
VOLUME_SPIKE_LOOKBACK_CANDLES = _get_int("VOLUME_SPIKE_LOOKBACK_CANDLES", 2)

# --- Filter ---
MIN_VOLUME_USDT = _get_float("MIN_VOLUME_USDT", 1_000_000)

# --- Binance / CCXT ---
EXCHANGE_ID = "binance"
EXCHANGE_OPTIONS = {"defaultType": "spot"}

# --- BTC Market Filter ---
# BTC phải uptrend (EMA50 > EMA200, price > EMA50) trước khi scan altcoin
BTC_SYMBOL = "BTC/USDT"
ENABLE_BTC_FILTER = _get_env("ENABLE_BTC_FILTER", "true").lower() not in ("false", "0", "no")

# --- Indicator params (có thể đổi qua env sau nếu cần) ---
EMA_PERIODS = (20, 50, 200)
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_SMA_PERIOD = 20           # Chuyển từ 65 xuống 20 để volume trung bình bám sát thực tế hơn
ACCUMULATION_LOOKBACK = 30
ACCUMULATION_RANGE_MAX_PCT = 35.0 # Mở rộng biên độ tích lũy cho crypto (15% là quá hẹp)
ACCUMULATION_POSITION_PCT = 50.0 # Giá nến trước nằm ở nửa trên của vùng tích lũy (áp lực mua gom sát kháng cự)
BREAKOUT_THRESHOLD_PCT = 98.0    # breakout khi giá >= 98% kháng cự
VOLUME_SPIKE_MIN_RATIO = 2.5     # 2.5x SMA(volume) là đủ xác nhận dòng tiền, 6x thường là FOMO đu đỉnh
RSI_MIN_DAILY = 50               # Phe mua kiểm soát hoàn toàn
ATR_SL_MULTIPLIER = 2.5          # Nới Stop Loss tránh bị quét râu nến

# --- Candles ---
CANDLES_1D = 250
# CANDLES_1H cần đủ để SMA warmup (65) + buffer thực chiến; 200 = 65 + 135 candle hữu ích
CANDLES_1H = 200

# --- Scan throttle ---
# Delay (giây) giữa mỗi symbol để tránh hit Binance rate limit (50 symbols × 2 timeframes)
SCAN_DELAY_SECONDS = _get_float("SCAN_DELAY_SECONDS", 0.3)

# --- Logging ---
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "signal_bot.log"
LOG_LEVEL = _get_env("LOG_LEVEL", "INFO").upper()
