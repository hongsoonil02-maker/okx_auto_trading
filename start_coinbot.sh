#!/bin/bash
DIR="/home/hongsoonil02/quant_system"
cd "$DIR" || exit 1

echo "[$(date)] coinbot cleanup/start 시작 (systemd 기반)"

# 1) 기존 프로세스 정리 (실존하는 파일만)
pkill -9 -f 'master_bot_orchestrator.py' >/dev/null 2>&1 || true
pkill -9 -f 'bot_c_okx_swap.py' >/dev/null 2>&1 || true
pkill -9 -f 'okx_major_strategy.py' >/dev/null 2>&1 || true
pkill -9 -f 'okx_venture_strategy.py' >/dev/null 2>&1 || true
pkill -9 -f 'daily_analyzer.py' >/dev/null 2>&1 || true

for port in 8000 8001 8005 8013; do
  fuser -k -9 "${port}/tcp" >/dev/null 2>&1 || true
done
sleep 2

# 2) stale pidfile 정리
rm -f bot_c_okx_swap.pid || true

# 3) Systemd 서비스들 시작
echo "Systemd 기반 봇들을 시작합니다..."
systemctl --user daemon-reload
systemctl --user start master_bot_orchestrator bot_c_okx_swap okx_major_strategy okx_venture_strategy

sleep 2
echo "✅ Coinbot 프로세스 구동 완료!"
ss -tlnp | grep -E ':(8000|8001|8013)' || true
