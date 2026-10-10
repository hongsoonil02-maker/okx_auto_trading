#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$DIR/kis_nasdaq_jev_trader.pid"

if [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null; then
    echo "⚠️ kis_nasdaq_jev_trader is already running with PID $(cat "$PIDFILE")"
    exit 0
fi

nohup /usr/bin/python3 "$DIR/kis_nasdaq_jev_trader.py" > /dev/null 2>&1 &
echo $! > "$PIDFILE"
echo "✅ Started kis_nasdaq_jev_trader with PID $(cat "$PIDFILE")"
