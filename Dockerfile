# Optional: chạy bot trong Docker (Python 3.10+)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config.py main.py signal_detector.py telegram_bot.py ./

# .env phải mount hoặc truyền qua -e
CMD ["python", "main.py"]
