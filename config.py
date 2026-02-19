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
POLLING_INTERVAL_HOURS = _get_int("POLLING_INTERVAL_HOURS", 4)

# --- Filter ---
MIN_VOLUME_USDT = _get_float("MIN_VOLUME_USDT", 1_000_000)

# --- Binance / CCXT ---
EXCHANGE_ID = "binance"
EXCHANGE_OPTIONS = {"defaultType": "spot"}

# --- Indicator params (có thể đổi qua env sau nếu cần) ---
EMA_PERIODS = (20, 50, 200)
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_SMA_PERIOD = 65
SUPPORT_RESISTANCE_PERIOD = 20
ACCUMULATION_LOOKBACK = 30
ACCUMULATION_RANGE_MAX_PCT = 15.0
ACCUMULATION_BOTTOM_PCT = 30.0   # giá gần đáy = trong 30% dưới của range
BREAKOUT_THRESHOLD_PCT = 98.0    # breakout khi giá >= 98% kháng cự
VOLUME_SPIKE_MIN_RATIO = 2.5    # 250% = 2.5x
RSI_MIN_DAILY = 40

# --- Candles ---
CANDLES_1D = 250
CANDLES_4H = 100
CANDLES_1H = 100

# --- Logging ---
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "signal_bot.log"
LOG_LEVEL = _get_env("LOG_LEVEL", "INFO").upper()
