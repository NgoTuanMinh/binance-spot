# Signal Bot – Binance + Telegram

Bot Python quét tín hiệu BUY (swing) từ Binance qua CCXT, phân tích 3 lớp (xu hướng, volume spike, breakout tích lũy), gửi message chi tiết lên Telegram. **Không tự động giao dịch.**

## Yêu cầu

- Python 3.10+
- Ubuntu 22.04 (hoặc tương đương)

## Cài đặt nhanh (Ubuntu)

```bash
cd /path/to/Binance-spot
bash setup_ubuntu.sh
```

Chỉnh `.env` (bắt buộc):

- `TELEGRAM_BOT_TOKEN` – token từ [@BotFather](https://t.me/BotFather)
- `TELEGRAM_CHAT_ID` – ID nhóm/chat nhận tin (có thể lấy từ [@userinfobot](https://t.me/userinfobot))

## Chạy thử

```bash
source .venv/bin/activate
python main.py
```

## Chạy bằng PM2

```bash
source .venv/bin/activate
pm2 start main.py --name signal-bot --interpreter .venv/bin/python
pm2 save && pm2 startup
```

Hoặc:

```bash
pm2 start "python main.py" --name signal-bot --cwd /path/to/Binance-spot
```

## Cấu hình (.env)

| Biến | Mô tả | Mặc định |
|------|--------|----------|
| `TELEGRAM_BOT_TOKEN` | Token bot Telegram | (bắt buộc) |
| `TELEGRAM_CHAT_ID` | Chat/group nhận tin | (bắt buộc) |
| `SYMBOLS` | Cặp quét (cách nhau dấu phẩy). **Để trống** = dùng top N theo volume 24h | (trống) |
| `TOP_SYMBOLS_COUNT` | Số symbol lấy theo volume 24h khi SYMBOLS trống (10–100) | `50` |
| `POLLING_INTERVAL_HOURS` | Chu kỳ quét (giờ) | `4` |
| `MIN_VOLUME_USDT` | Volume 24h tối thiểu (USDT) khi lọc top symbol | `1000000` |
| `LOG_LEVEL` | DEBUG / INFO / WARNING / ERROR | `INFO` |

## Cấu trúc

- `main.py` – Entry, scheduling (APScheduler), gọi quét và gửi Telegram
- `signal_detector.py` – Logic 3 lớp: trend daily, volume spike, accumulation breakout
- `telegram_bot.py` – Format và gửi message Telegram
- `config.py` – Đọc từ `.env`
- `symbols.py` – Lấy danh sách symbol (từ SYMBOLS hoặc top 10–100 theo volume 24h)
- `logs/signal_bot.log` – Log file

## Chiến lược (tóm tắt)

1. **Lớp 1 (Daily):** Giá > EMA50, EMA50 > EMA200, RSI(14) > 40  
2. **Lớp 2:** Volume hiện tại > 250% SMA(volume, 65)  
3. **Lớp 3:** Vùng tích lũy 30 nến (range < 15%), giá từng ở đáy (30% dưới), breakout ≥ 98% kháng cự  

Khi đủ 3 lớp → gửi Telegram với Entry, SL, TP1 (10%), TP2 (20%), Risk %, R:R, Volume ratio, RSI, ATR%.

## Lưu ý

- Không đặt lệnh tự động; chỉ gửi tín hiệu để bạn tự quyết định.
- DYOR – đây chỉ là tín hiệu, không phải lời khuyên tài chính.
