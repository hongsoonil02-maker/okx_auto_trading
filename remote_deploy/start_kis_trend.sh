#!/bin/bash
# start_kis_trend.sh — Start KIS Trend Following Daemon
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$CURRENT_DIR/kis_trend_trader.pid"
LOG_FILE="$CURRENT_DIR/kis_trend_trader.log"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "⚠️ KIS Trend Trader is already running (PID: $PID)"
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

echo "🚀 Starting KIS Trend Trader (KRX Trend Following Engine)..."
nohup /usr/bin/python3 "$CURRENT_DIR/kis_trend_trader.py" > /dev/null 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo "✅ Started successfully with PID: $NEW_PID"
echo "📄 Log file: $LOG_FILE"
