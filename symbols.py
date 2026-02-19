"""
Lấy danh sách symbol để quét: từ .env (SYMBOLS) hoặc top N theo khối lượng 24h từ Binance.
"""
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Giới hạn theo spec: top 10 -> 100
TOP_SYMBOLS_MIN = 10
TOP_SYMBOLS_MAX = 100


def get_top_symbols_by_volume(
    exchange: Any,
    top_n: int,
    min_volume_usdt: float,
    max_retries: int = 3,
) -> list[str]:
    """
    Lấy top N cặp USDT có khối lượng giao dịch 24h (quote volume USDT) cao nhất.
    Chỉ lấy symbol có volume >= min_volume_usdt.
    """
    top_n = max(TOP_SYMBOLS_MIN, min(TOP_SYMBOLS_MAX, top_n))
    for attempt in range(max_retries):
        try:
            tickers = exchange.fetch_tickers()
            # Chọn cặp *USDT, lấy quote volume (USDT)
            candidates = []
            for symbol, data in tickers.items():
                if not symbol.endswith("/USDT"):
                    continue
                quote_vol = float(data.get("quoteVolume") or data.get("quote_volume") or 0)
                if quote_vol < min_volume_usdt:
                    continue
                candidates.append((symbol, quote_vol))
            candidates.sort(key=lambda x: x[1], reverse=True)
            symbols = [s[0] for s in candidates[:top_n]]
            logger.info(
                "Fetched top %d symbols by 24h volume (min volume USDT=%.0f), got %d",
                top_n, min_volume_usdt, len(symbols),
            )
            return symbols
        except Exception as e:
            wait = (2 ** attempt) + 1
            logger.warning("fetch_tickers attempt %d failed: %s, retry in %ds", attempt + 1, e, wait)
            if attempt == max_retries - 1:
                raise
            time.sleep(wait)
    return []
