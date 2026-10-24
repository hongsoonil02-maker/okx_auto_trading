#!/bin/bash
# stop_kis_trend.sh — Stop KIS Trend Following Daemon
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$CURRENT_DIR/kis_trend_trader.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "🛑 Stopping KIS Trend Trader (PID: $PID)..."
        kill "$PID"
        sleep 1
        if ps -p "$PID" > /dev/null 2>&1; then
            kill -9 "$PID"
        fi
        echo "✅ Stopped successfully."
    else
        echo "⚠️ Process $PID is not running."
    fi
    rm -f "$PID_FILE"
else
    echo "⚠️ No PID file found. Checking ps aux..."
    pkill -f "kis_trend_trader.py"
    echo "✅ Checked and cleaned up any running processes."
fi
