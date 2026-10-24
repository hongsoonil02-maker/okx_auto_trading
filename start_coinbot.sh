#!/bin/bash
DIR="/home/hongsoonil02/quant_system"
cd "$DIR" || exit 1

echo "=========================================================================="
echo "[$(date)] Main Account (Major + Venture Dual Engine) 가동 시작"
echo "=========================================================================="

# 1) Preflight Self-Test
echo "🔍 [1/3] 봇 사전 자가진단(Preflight Self-Test) 실행 중..."
python3 "$DIR/preflight_check.py" --env "$DIR/.env"
if [ $? -ne 0 ]; then
  echo "🚨 Pre-flight 진단 실패로 봇 가동을 즉시 중단합니다. 위 에러를 확인하세요."
  exit 1
fi

# 2) 잔여 프로세스 정리 (메인 전용 포트 8010, 8015만 정리 - 서브계정 보호)
echo "🧹 [2/3] 기존 잔여 프로세스 및 포트 정리 중..."
pkill -9 -f "$DIR/master_bot_orchestrator.py" >/dev/null 2>&1 || true
pkill -9 -f "$DIR/bot_c_okx_swap.py" >/dev/null 2>&1 || true
pkill -9 -f "$DIR/okx_major_strategy.py" >/dev/null 2>&1 || true
pkill -9 -f "$DIR/okx_venture_strategy.py" >/dev/null 2>&1 || true

for port in 8010 8015; do
  fuser -k -9 "${port}/tcp" >/dev/null 2>&1 || true
done
rm -f bot_c_okx_swap.pid || true
sleep 1

# 3) Systemd 서비스 시작
echo "🚀 [3/3] Main Dual Strategy (Major + Venture), Orchestrator 및 Bot C 시작 중..."
systemctl --user daemon-reload
systemctl --user start bot_c_okx_swap master_bot_orchestrator okx_major_strategy okx_venture_strategy

sleep 2
echo "✅ Main Account (Major + Venture 1+2) 구동 완료!"
ss -tlnp | grep -E ':(8010|8015)' || true
