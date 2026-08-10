#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trade_metrics.py — state/trades.jsonl 체결 기록 기반 실현손익 리포트
심볼+방향별 평단 추적으로 실현 손익 근사 계산 (수수료 테이커 0.05% 편도 반영)
사용: python3 trade_metrics.py [days]
"""
import json
import os
import sys
import time
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADES_FILE = os.path.join(BASE_DIR, "state", "trades.jsonl")
FEE = 0.0005  # 편도 테이커


def load(days=None):
    if not os.path.exists(TRADES_FILE):
        return []
    cutoff = time.time() - days * 86400 if days else 0
    out = []
    with open(TRADES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ts", 0) >= cutoff and r.get("price"):
                out.append(r)
    return out


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else None
    trades = load(days)
    if not trades:
        print("기록된 체결 없음 (state/trades.jsonl)")
        return

    # (symbol, pos_side) -> {'qty', 'avg'}
    pos = defaultdict(lambda: {"qty": 0.0, "avg": 0.0})
    realized = []
    fees = 0.0

    def pos_side_of(r):
        return "short" if r["side"] in ("SELL", "CLOSE_SHORT") else "long"

    for r in trades:
        key = (r["symbol"], pos_side_of(r))
        p = pos[key]
        px, amt = float(r["price"]), float(r["amount"])
        is_open = r["side"] in ("BUY", "SELL")

        if is_open:
            fees += amt * px * FEE
            new_qty = p["qty"] + amt
            p["avg"] = (p["avg"] * p["qty"] + px * amt) / new_qty if new_qty else 0
            p["qty"] = new_qty
        else:
            close_qty = p["qty"] if amt == 0 else min(amt, p["qty"])
            if close_qty <= 0 or p["qty"] <= 0:
                continue
            fees += close_qty * px * FEE
            sgn = 1 if key[1] == "long" else -1
            pnl_usdt = (px - p["avg"]) * close_qty * sgn
            realized.append({
                "symbol": key[0], "side": key[1], "qty": close_qty,
                "entry": p["avg"], "exit": px, "pnl_usdt": pnl_usdt, "ts": r["ts"],
            })
            p["qty"] -= close_qty
            if p["qty"] <= 1e-9:
                p["qty"], p["avg"] = 0.0, 0.0

    print("=" * 66)
    print(f"📊 실현손익 리포트 ({'최근 %d일' % days if days else '전체'}) | 체결 {len(trades)}건")
    print("=" * 66)
    if not realized:
        print("청산된 거래 없음 (미실현 포지션만 존재)")
    else:
        wins = [t for t in realized if t["pnl_usdt"] > 0]
        total = sum(t["pnl_usdt"] for t in realized)
        wr = len(wins) / len(realized) * 100
        gp = sum(t["pnl_usdt"] for t in wins)
        gl = -sum(t["pnl_usdt"] for t in realized if t["pnl_usdt"] <= 0)
        pf = gp / gl if gl > 0 else float("inf")
        print(f"청산 {len(realized)}건 | 승률 {wr:.1f}% | 실현손익 {total:+.2f} USDT | "
              f"PF {pf:.2f} | 수수료 추정 {fees:.2f} USDT")
        by_sym = defaultdict(float)
        for t in realized:
            by_sym[t["symbol"]] += t["pnl_usdt"]
        print("\n[심볼별 실현손익]")
        for sym, pnl in sorted(by_sym.items(), key=lambda x: x[1]):
            print(f"  {sym:<20} {pnl:+10.2f} USDT")

    open_pos = {k: v for k, v in pos.items() if v["qty"] > 0}
    if open_pos:
        print("\n[미청산 포지션 (평단)]")
        for (sym, side), v in open_pos.items():
            print(f"  {sym:<20} {side:<5} qty={v['qty']:.4f} avg={v['avg']:.2f}")


if __name__ == "__main__":
    main()
