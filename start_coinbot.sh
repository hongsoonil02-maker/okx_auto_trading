#!/bin/bash
DIR="/home/hongsoonil02/quant_system"
cd "$DIR" || exit 1

echo "[$(date)] coinbot cleanup/start 백그라운드 가동 시작"

# 1) 기존 프로세스 정리
pkill -9 -f 'master_bot_orchestrator.py' >/dev/null 2>&1 || true
pkill -9 -f 'bot_a_kr_stock.py' >/dev/null 2>&1 || true
pkill -9 -f 'bot_c_okx_swap.py' >/dev/null 2>&1 || true
pkill -9 -f 'daily_analyzer.py' >/dev/null 2>&1 || true

for port in 8000 8001 8003 8005; do
  fuser -k -9 "${port}/tcp" >/dev/null 2>&1 || true
done
sleep 2

# 2) stale pidfile 정리
rm -f bot_a_kr_stock.pid bot_c_okx_swap.pid || true

# 3) 사용자 권한 백그라운드 봇 실행
nohup python3 bot_a_kr_stock.py > bot_a_kr_stock.log 2>&1 &
nohup python3 bot_c_okx_swap.py 8003 > bot_c_okx_swap.log 2>&1 &

sleep 3

# 4) 마스터 오케스트레이터 가동
nohup python3 master_bot_orchestrator.py > master_orchestrator.log 2>&1 &

sleep 2
echo "✅ Coinbot 프로세스 구동 완료!"
ss -tlnp | grep -E ':(8000|8001|8003)' || true
