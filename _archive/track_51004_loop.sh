#!/bin/bash
# 51004 클리핑 효과 24시간 추적 — 15분마다 state/51004_clip_effect.log에 기록
DIR="/home/hongsoonil02/$1"
LOG="$DIR/state/51004_clip_effect.log"
while true; do
  python3 - "$DIR" "$LOG" <<'PY'
import sys, json, datetime, os
d = sys.argv[1]; log = sys.argv[2]
ts = datetime.datetime.utcnow().strftime("%m/%d %H:%M UTC")
# 베이스라인 이후 거래
trades = []
p = f"{d}/state/trades.jsonl"
if os.path.exists(p):
    base = 1788080040
    for line in open(p):
        try:
            t = json.loads(line)
            if t.get("ts",0) >= base:
                trades.append(t)
        except: pass
# 51004 에러 수 (전체 누적)
cnt = 0
bp = f"{d}/bot_c_okx_swap.log"
if os.path.exists(bp):
    cnt = sum(1 for line in open(bp) if "51004" in line)
entry = sum(1 for t in trades if t["side"] in ("BUY","SELL"))
closes = sum(1 for t in trades if t["side"] in ("CLOSE_LONG","CLOSE_SHORT") or t.get("amount",0)>0)
line = f"{ts} | entries={entry} closes={closes} 51004_total_errors={cnt}\n"
with open(log, "a") as f:
    f.write(line)
print(line, end="")
PY
  sleep 900  # 15분
done
