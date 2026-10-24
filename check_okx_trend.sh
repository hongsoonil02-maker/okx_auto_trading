#!/usr/bin/env bash
echo "======================================================================"
echo "📊 OKX Quant Trend Trader v2.0 Status Monitor"
echo "======================================================================"
echo ""
echo "1. Service Status:"
systemctl --user status okx_trend_trader.service --no-pager | head -n 12

echo ""
echo "2. Active Positions in State File:"
if [ -f "/home/hongsoonil02/quant_system/state/okx_trend_positions.json" ]; then
    cat /home/hongsoonil02/quant_system/state/okx_trend_positions.json
else
    echo "No state file found."
fi

echo ""
echo "3. Recent Log Output (Last 20 lines):"
tail -n 20 /home/hongsoonil02/quant_system/okx_trend_trader.log
echo "======================================================================"
