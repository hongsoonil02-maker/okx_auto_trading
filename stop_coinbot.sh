#!/bin/bash
DIR="/home/hongsoonil02/quant_system"
cd "$DIR" || exit 1

echo "[$(date)] coinbot 중지 시작 (systemd 기반)"

# 1) systemd 서비스 중지 및 비활성화
systemctl --user stop okx_major_strategy okx_venture_strategy master_bot_orchestrator bot_c_okx_swap 2>/dev/null || true
systemctl --user disable okx_major_strategy okx_venture_strategy master_bot_orchestrator bot_c_okx_swap 2>/dev/null || true

# 2) 잔여 프로세스 강제 종료
pkill -9 -f 'master_bot_orchestrator.py' >/dev/null 2>&1 || true
pkill -9 -f 'bot_c_okx_swap.py' >/dev/null 2>&1 || true
pkill -9 -f 'okx_major_strategy.py' >/dev/null 2>&1 || true
pkill -9 -f 'okx_venture_strategy.py' >/dev/null 2>&1 || true

for port in 8000 8001 8005 8009 8013; do
  fuser -k -9 "${port}/tcp" >/dev/null 2>&1 || true
done

# 3) stale pidfile 정리
rm -f bot_c_okx_swap.pid || true

echo "✅ 모든 코인 자동매매 봇이 정상적으로 중지되었습니다."
