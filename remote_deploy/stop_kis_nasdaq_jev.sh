#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$DIR/kis_nasdaq_jev_trader.pid"

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "🛑 Stopped kis_nasdaq_jev_trader (PID $PID)"
    fi
    rm -f "$PIDFILE"
else
    pkill -f "kis_nasdaq_jev_trader.py"
    echo "🛑 Stopped kis_nasdaq_jev_trader (pkill)"
fi
