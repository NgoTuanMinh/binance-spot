"""
Gửi message tín hiệu BUY lên Telegram (HTML, emoji).
Chỉ gửi khi có tín hiệu BUY, không gửi HOLD.
"""
import asyncio
import logging
from datetime import datetime

from telegram import Bot

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from signal_detector import BuySignal

logger = logging.getLogger(__name__)


def _escape_html(text: str) -> str:
    """Escape cho HTML Telegram (chỉ & < >)."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def format_buy_message(signal: BuySignal) -> str:
    """
    Format nội dung message theo spec: HTML với emoji.
    Telegram HTML: <code>, <b>, <i>, <pre>, <a href="...">.
    """
    t = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    entry = signal.entry
    sl = signal.stop_loss
    tp1 = signal.tp1
    tp2 = signal.tp2

    lines = [
        "🚨 🚀 BUY SIGNAL DETECTED 🚨",
        "",
        f"🪙 Symbol: {_escape_html(signal.symbol)}",
        f"⏰ Time: {t}",
        "",
        "━━━━━━━━━━━━━━━━━━━━━",
        "📊 PRICE LEVELS",
        "━━━━━━━━━━━━━━━━━━━━━",
        f"💵 Entry: ${entry:.2f}",
        f"🛑 Stop Loss: ${sl:.2f}",
        f"🎯 TP1 (10%): ${tp1:.2f}",
        f"🎯 TP2 (20%): ${tp2:.2f}",
        "",
        "━━━━━━━━━━━━━━━━━━━━━",
        "📈 RISK METRICS",
        "━━━━━━━━━━━━━━━━━━━━━",
        f"⚠️ Risk: {signal.risk_pct:.1f}%",
        f"💰 Reward (TP1): {signal.reward_tp1_pct:.1f}%",
        f"⚖️ R:R Ratio: 1:{signal.rr_ratio:.1f}",
        "",
        "━━━━━━━━━━━━━━━━━━━━━",
        "🔍 SIGNAL STRENGTH",
        "━━━━━━━━━━━━━━━━━━━━━",
        f"📊 Volume Spike: {signal.volume_ratio:.1f}x avg",
        f"📦 Accumulation Range: {signal.accumulation_range_pct:.1f}%",
        f"📈 RSI (Daily): {signal.rsi_daily:.1f}",
        f"🌊 ATR (1h): {signal.atr_pct_1h:.1f}%",
        "",
        "━━━━━━━━━━━━━━━━━━━━━",
        "⚠️ DYOR - This is just signal, not financial advice",
    ]
    return "\n".join(lines)


class TelegramSender:
    """Gửi tin nhắn tín hiệu lên Telegram (async)."""

    def __init__(self, token: str | None = None, chat_id: str | None = None):
        self.token = token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self._bot = Bot(token=self.token) if self.token else None

    async def send_buy_signal(self, signal: BuySignal) -> bool:
        """Gửi 1 message tín hiệu BUY. HTML không dùng parse_mode HTML cho block code nên dùng pre hoặc text thuần."""
        if not self._bot or not self.chat_id:
            logger.error("Telegram not configured: missing token or chat_id")
            return False
        text = format_buy_message(signal)
        # Telegram API: HTML mode không hỗ trợ emoji block đặc biệt, gửi dạng text vẫn hiển thị emoji
        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=None,  # plain text để emoji và ký tự đặc biệt giữ nguyên
            )
            logger.info("Sent BUY signal to Telegram: %s", signal.symbol)
            return True
        except Exception as e:
            logger.exception("Telegram send failed: %s", e)
            return False


async def send_signal_async(signal: BuySignal) -> bool:
    """Helper: gửi 1 signal (dùng từ main sync)."""
    sender = TelegramSender()
    return await sender.send_buy_signal(signal)


def send_signal_sync(signal: BuySignal) -> bool:
    """Gửi tín hiệu từ code sync (main loop)."""
    return asyncio.run(send_signal_async(signal))
