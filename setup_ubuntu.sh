#!/usr/bin/env bash
# Setup script cho Ubuntu 22.04 - Signal Bot (Binance + Telegram)
# Chạy: bash setup_ubuntu.sh
# Sau khi setup, chạy bot bằng: pm2 start main.py --name signal-bot --interpreter python3

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/4] Checking Python 3.10+..."
if ! command -v python3 &>/dev/null; then
    echo "Python3 not found. Install: sudo apt update && sudo apt install -y python3 python3-pip python3-venv"
    exit 1
fi
PYVER=$(python3 -c "import sys; print(sys.version_info.major, sys.version_info.minor)")
echo "  Python: $(python3 --version)"

echo "[2/4] Creating virtualenv..."
python3 -m venv .venv
source .venv/bin/activate

echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[4/4] Environment file..."
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "  Created .env from .env.example - PLEASE EDIT .env and set TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID"
else
    echo "  .env already exists, skipping"
fi

echo ""
echo "Setup done. Next steps:"
echo "  1. Edit .env: nano .env  (set TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)"
echo "  2. Test run:  source .venv/bin/activate && python main.py"
echo "  3. Run with pm2: pm2 start main.py --name signal-bot --interpreter .venv/bin/python"
echo "     Or: pm2 start 'python main.py' --name signal-bot --cwd $SCRIPT_DIR"
echo ""
