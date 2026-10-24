#!/bin/bash
# check_ross_cameron.sh — Status & Monitoring for Ross Cameron Swing Engine
echo "=========================================================================="
echo "📊 [OKX Ross Cameron Swing Engine] 서비스 및 계좌 현황"
echo "=========================================================================="

echo "• 서비스 상태 (systemd --user):"
systemctl --user status okx_ross_cameron_swing.service --no-pager | head -n 12

echo ""
echo "• 실시간 포지션 파일 (ross_cameron_positions.json):"
if [ -f /home/hongsoonil02/quant_system/state/ross_cameron_positions.json ]; then
    cat /home/hongsoonil02/quant_system/state/ross_cameron_positions.json
else
    echo "  (보유 포지션 없음)"
fi

echo ""
echo "• 최근 엔진 로그 (마지막 15줄):"
tail -n 15 /home/hongsoonil02/quant_system/okx_ross_cameron_swing.log

echo "=========================================================================="
