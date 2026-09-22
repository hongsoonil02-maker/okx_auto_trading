#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_trades_by_sector.py — 섹터별/종목별 실현손익 분석"""
import asyncio, json, datetime
from collections import defaultdict
import ccxt.async_support as ccxt_async

MAJORS = {'BTC','ETH','SOL','XRP','ADA','AVAX','LINK','DOT','BNB','TRX'}
STOCKS = {'TSLA','NVDA','AAPL','AMZN','MSFT','META','GOOG','GOOGL','COIN','SPCX','OPENAI','ANTHROPIC',
          'RDDT','MU','SNDK','SOXL','XAU','CL','SKHY','KORU','CBRS','AEON','PLTR','AMD','INTC','QCOM',
          'BABA','UBER','ABNB','SNAP','MSTR','HOOD','RIVN','NIO','PYPL','SQ','SHOP','SPY','QQQ','IWM',
          'DIA','GLD','SLV','XAG','AXTI','CRCL','UNITREE','XIAOMI','LITE','UB','KR200','ISRG',
          'MRVL','SOXS','SKUU','HOME'}
MEMES = {'DOGE','SHIB','PEPE','BOME','WIF','BONK','FLOKI','FARTCOIN','PUMP','PEOPLE','MOODENG','PNUT',
         'ACT','NEIRO','TURBO','MEW','POPCAT','GIGA','BRETT','TRUMP','MELANIA','PENGU','AI16Z'}


def classify(base, lt_ms, first_trade_s):
    if base in STOCKS:
        return "주식연계"
    if base in MAJORS:
        return "메이저"
    if lt_ms and first_trade_s and (first_trade_s * 1000 - lt_ms) < 60 * 86400 * 1000:
        return "신규상장"
    if base in MEMES:
        return "밈"
    return "알트"


async def main():
    ex = ccxt_async.okx({"options": {"defaultType": "swap"}})
    await ex.load_markets()
    list_time = {}
    for sym, m in ex.markets.items():
        if m.get('swap'):
            try:
                list_time[sym] = int((m.get('info') or {}).get('listTime') or 0)
            except Exception:
                list_time[sym] = 0

    rows = []
    with open("state/trades.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    cut = datetime.datetime(2026, 8, 16, 0, 0).timestamp()
    pos, pnls = {}, []
    sym_stats = defaultdict(lambda: {'pnl': 0.0, 'wins': 0, 'losses': 0})
    first_seen, n_ent, n_cls = {}, 0, 0
    for r in rows:
        ts = r.get('ts', 0)
        if ts < cut:
            continue
        sym, side = r['symbol'], r['side']
        px, amt = float(r.get('price') or 0), float(r.get('amount') or 0)
        if not px:
            continue
        cs = float((ex.markets.get(sym) or {}).get('contractSize') or 1)
        first_seen.setdefault(sym, ts)
        sk = "short" if side in ("SELL", "CLOSE_SHORT") else "long"
        key = (sym, sk)
        qty, avg = pos.get(key, (0.0, 0.0))
        if side in ("BUY", "SELL"):
            n_ent += 1
            nq = qty + amt
            pos[key] = (nq, (avg * qty + px * amt) / nq if nq else 0.0)
        else:
            n_cls += 1
            cq = qty if amt == 0 else min(amt, qty)
            if cq > 0 and avg > 0:
                sgn = 1 if sk == "long" else -1
                p = (px - avg) * cq * cs * sgn
                pnls.append(p)
                st = sym_stats[sym]
                st['pnl'] += p
                st['wins' if p > 0 else 'losses'] += 1
            nq = qty - cq
            pos[key] = (max(nq, 0.0), avg if nq > 1e-9 else 0.0)

    total = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    print(f"실계정 가동 (08-16~): 진입{n_ent} 청산{n_cls} | 실현 {total:+,.0f} USDT | "
          f"승률 {wins}/{len(pnls)} ({wins / max(1, len(pnls)) * 100:.0f}%)")

    sec_stats = defaultdict(lambda: [0.0, 0])
    sym_sec = {}
    for sym, st in sym_stats.items():
        base = sym.split('/')[0]
        sec = classify(base, list_time.get(sym, 0), first_seen.get(sym, 0))
        sym_sec[sym] = sec
        sec_stats[sec][0] += st['pnl']
        sec_stats[sec][1] += st['wins'] + st['losses']
    print("\n【섹터별 실현손익】")
    for sec, (pnl, n) in sorted(sec_stats.items(), key=lambda x: -x[1][0]):
        print(f"  {sec:5s}: {pnl:+12,.2f} USDT ({n}회)")

    ranked = sorted(sym_stats.items(), key=lambda x: -x[1]['pnl'])
    print(f"\n【종목별 전체 {len(ranked)}개】 수익 상위 15:")
    for sym, st in ranked[:15]:
        w, l = st['wins'], st['losses']
        print(f"  {sym:24s}[{sym_sec[sym]:4s}] {st['pnl']:>+11,.2f}  ({w}승{l}패)")
    print("  ... 손실 하위 10:")
    for sym, st in ranked[-10:]:
        w, l = st['wins'], st['losses']
        print(f"  {sym:24s}[{sym_sec[sym]:4s}] {st['pnl']:>+11,.2f}  ({w}승{l}패)")

    # 섹터별 대표 종목 통계
    print("\n【섹터 상세】")
    for sec in ("메이저", "알트", "밈", "신규상장", "주식연계"):
        syms = [(s, st) for s, st in ranked if sym_sec[s] == sec]
        if not syms:
            continue
        pnl = sum(st['pnl'] for _, st in syms)
        w = sum(st['wins'] for _, st in syms)
        l = sum(st['losses'] for _, st in syms)
        best = max(syms, key=lambda x: x[1]['pnl'])
        worst = min(syms, key=lambda x: x[1]['pnl'])
        print(f"  ▸ {sec}: {len(syms)}종목 | {pnl:+,.0f} USDT | {w}승{l}패 ({w/max(1,w+l)*100:.0f}%)")
        print(f"      최고 {best[0].split('/')[0]} {best[1]['pnl']:+,.0f} / 최악 {worst[0].split('/')[0]} {worst[1]['pnl']:+,.0f}")
    await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
