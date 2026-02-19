"""
Bot chính: polling Binance theo lịch (mỗi 4h), phát hiện tín hiệu BUY và gửi Telegram.
Không tự động giao dịch.
"""
import logging
import sys
from pathlib import Path

import ccxt  # pyright: ignore[reportMissingImports]
from apscheduler.schedulers.blocking import BlockingScheduler  # pyright: ignore[reportMissingImports]
from apscheduler.triggers.interval import IntervalTrigger  # pyright: ignore[reportMissingImports]

from config import (
    EXCHANGE_ID,
    EXCHANGE_OPTIONS,
    LOG_FILE,
    LOG_LEVEL,
    LOG_DIR,
    MIN_VOLUME_USDT,
    POLLING_INTERVAL_HOURS,
    SYMBOLS,
    TOP_SYMBOLS_COUNT,
    USE_TOP_BY_VOLUME,
)
from signal_detector import SignalDetector
from symbols import get_top_symbols_by_volume
from telegram_bot import send_signal_sync

# --- Logging ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=log_fmt,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def create_exchange():
    """Tạo instance CCXT Binance spot."""
    return getattr(ccxt, EXCHANGE_ID)(EXCHANGE_OPTIONS)


def get_symbols_to_scan(exchange):
    """Danh sách symbol cần quét: từ SYMBOLS hoặc top N theo volume 24h."""
    if USE_TOP_BY_VOLUME:
        return get_top_symbols_by_volume(exchange, TOP_SYMBOLS_COUNT, MIN_VOLUME_USDT)
    return SYMBOLS


def run_scan():
    """Quét toàn bộ symbol (từ config hoặc top volume), gửi Telegram khi có tín hiệu BUY."""
    exchange = create_exchange()
    symbols = get_symbols_to_scan(exchange)
    logger.info("Scanning %d symbols: %s", len(symbols), symbols[:15] if len(symbols) > 15 else symbols)
    detector = SignalDetector(exchange)
    for symbol in symbols:
        try:
            signal = detector.check_buy_signal(symbol)
            if signal:
                logger.info("BUY signal: %s entry=%.2f", signal.symbol, signal.entry)
                ok = send_signal_sync(signal)
                if not ok:
                    logger.warning("Failed to send Telegram for %s", signal.symbol)
            else:
                logger.debug("No signal for %s", symbol)
        except Exception as e:
            logger.exception("Error scanning %s: %s", symbol, e)
            # Tiếp tục symbol tiếp theo, không crash
    try:
        exchange.close()
    except Exception:
        pass


def main():
    """Chạy scheduler: mỗi POLLING_INTERVAL_HOURS chạy run_scan một lần."""
    mode = f"top {TOP_SYMBOLS_COUNT} by 24h volume" if USE_TOP_BY_VOLUME else f"fixed list ({len(SYMBOLS)} symbols)"
    logger.info(
        "Signal bot started. Polling every %s hour(s). Mode: %s",
        POLLING_INTERVAL_HOURS,
        mode,
    )
    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_scan,
        trigger=IntervalTrigger(hours=POLLING_INTERVAL_HOURS),
        id="scan",
    )
    # Chạy 1 lần ngay khi start
    run_scan()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
