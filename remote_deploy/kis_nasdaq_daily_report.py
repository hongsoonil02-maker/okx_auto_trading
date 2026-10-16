# -*- coding: utf-8 -*-
"""
kis_nasdaq_daily_report.py — KIS Nasdaq Trader 성과 리포트
kis_nasdaq_trader.log에서 매수/매도 이벤트를 FIFO 매칭해 실현 손익을 집계한다.
- 수수료: 편도 FEE_PCT_PER_SIDE (기본 0.25%, 우대 적용 시 조정)
- 사용법: python3 kis_nasdaq_daily_report.py [--days N]
"""
import os
import re
import sys
import json
from collections import defaultdict, deque
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(CURRENT_DIR, "kis_nasdaq_trader.log")
POSITIONS_FILE = os.path.join(CURRENT_DIR, "kis_nasdaq_positions.json")

FEE_PCT_PER_SIDE = float(os.getenv("KIS_FEE_PCT_PER_SIDE", "0.0025"))  # 편도 0.25%

RE_BUY_INTENT = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2},\d+ .*?\[Jev Signal Executing\] (\w+) .*?Px: \$([\d.]+) .*?Sizing: (\d+) shares"
)
RE_BUY_ACCEPTED = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2},\d+ .*?\[Order (?:Accepted|Success)\] BUY (\w+) x (\d+)"
)
RE_BUY_FAILED = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2},\d+ .*?\[Order (?:Rejected|Exception)\] BUY (\w+)"
)
RE_PARTIAL = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2},\d+ .*?\[Partial Take Profit\] (\w+):.*?Px: \$([\d.]+) \| Qty: (\d+)"
)
RE_EXIT = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2},\d+ .*?\[Exit Triggered\] (\w+) \| (\w+).*?Px: \$([\d.]+) \| Qty: (\d+)"
)


def parse_trades(days_filter=None):
    lots = defaultdict(deque)   # sym -> deque of [qty, price, date]
    closed = []                 # (date, sym, qty, buy_px, sell_px, reason)
    pending_buy = {}            # sym -> (date, px, qty) — 접수 확인 전 매수 시도

    if not os.path.exists(LOG_FILE):
        print(f"로그 파일 없음: {LOG_FILE}")
        return closed, lots

    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m_intent = RE_BUY_INTENT.search(line)
            if m_intent:
                date, sym, px, qty = m_intent.group(1), m_intent.group(2), float(m_intent.group(3)), int(m_intent.group(4))
                pending_buy[sym] = (date, px, qty)
                continue

            m_acc = RE_BUY_ACCEPTED.search(line)
            if m_acc:
                date, sym, acc_qty = m_acc.group(1), m_acc.group(2), int(m_acc.group(3))
                intent = pending_buy.pop(sym, None)
                if intent:
                    # 접수된 수량은 주문 수량과 동일 (부분체결은 원장 대사가 조정)
                    lots[sym].append([acc_qty, intent[1], intent[0]])
                else:
                    lots[sym].append([acc_qty, 0.0, date])  # 가격 미상 로트
                continue

            m_fail = RE_BUY_FAILED.search(line)
            if m_fail:
                pending_buy.pop(m_fail.group(2), None)  # 거절된 매수 시도 폐기
                continue

            m_part = RE_PARTIAL.search(line)
            if m_part:
                date, sym, px, qty = m_part.group(1), m_part.group(2), float(m_part.group(3)), int(m_part.group(4))
                _consume(lots[sym], qty, px, date, sym, "PARTIAL_TP", closed)
                continue

            m_exit = RE_EXIT.search(line)
            if m_exit:
                date, sym, reason, px, qty = m_exit.group(1), m_exit.group(2), m_exit.group(3), float(m_exit.group(4)), int(m_exit.group(5))
                _consume(lots[sym], qty, px, date, sym, reason, closed)

    if days_filter:
        cutoff_dates = sorted({c[0] for c in closed})[-days_filter:]
        closed = [c for c in closed if c[0] in cutoff_dates]
    return closed, lots


def _consume(queue, sell_qty, sell_px, date, sym, reason, closed):
    """FIFO로 매수 로트를 소진하며 왕복 손익 기록"""
    remaining = sell_qty
    while remaining > 0 and queue:
        lot = queue[0]
        take = min(lot[0], remaining)
        closed.append((date, sym, take, lot[1], sell_px, reason))
        lot[0] -= take
        remaining -= take
        if lot[0] == 0:
            queue.popleft()
    # 매수 기록 없는 매도(수동 매수분 등)는 집계에서 제외


def main():
    days = None
    if "--days" in sys.argv:
        days = int(sys.argv[sys.argv.index("--days") + 1])

    closed, open_lots = parse_trades(days)

    print("=" * 78)
    print("📊 KIS Nasdaq Trader 성과 리포트")
    print(f"   수수료 모델: 편도 {FEE_PCT_PER_SIDE*100:.2f}% (왕복 {FEE_PCT_PER_SIDE*2*100:.2f}%) | 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    if not closed:
        print("완결된 왕복 거래가 없습니다 (매수만 존재하거나 로그 없음).")
    else:
        by_date = defaultdict(list)
        for c in closed:
            by_date[c[0]].append(c)

        total_net_usd = 0.0
        total_buy_amt = 0.0
        all_nets = []
        wins, losses = 0, 0

        print(f"\n{'날짜(UTC)':<12}{'종목':<7}{'수량':>4}{'매수가':>10}{'매도가':>10}{'사유':<16}{'순손익%':>8}{'순손익$':>9}")
        print("-" * 78)
        for date in sorted(by_date):
            for _, sym, qty, buy_px, sell_px, reason in by_date[date]:
                gross_pct = (sell_px / buy_px - 1.0) * 100.0
                net_pct = gross_pct - FEE_PCT_PER_SIDE * 2 * 100.0
                buy_amt = qty * buy_px
                net_usd = buy_amt * net_pct / 100.0
                total_net_usd += net_usd
                total_buy_amt += buy_amt
                all_nets.append(net_pct)
                if net_pct > 0:
                    wins += 1
                else:
                    losses += 1
                print(f"{date:<12}{sym:<7}{qty:>4}{buy_px:>10.2f}{sell_px:>10.2f}{reason:<16}{net_pct:>+7.2f}%{net_usd:>+8.2f}")

        n = len(all_nets)
        win_rate = wins / n * 100.0 if n else 0.0
        avg_win = sum(x for x in all_nets if x > 0) / wins if wins else 0.0
        avg_loss = sum(x for x in all_nets if x <= 0) / losses if losses else 0.0
        print("-" * 78)
        print(f"\n📈 요약: 왕복 {n}건 | 승률 {win_rate:.1f}% ({wins}승 {losses}패)")
        print(f"   평균 익절: {avg_win:+.2f}% | 평균 손절: {avg_loss:+.2f}% | 손익비: {abs(avg_win/avg_loss) if avg_loss else 0:.2f}x")
        print(f"   누적 순손익(수수료 차감 후): ${total_net_usd:+,.2f} | 투입 원금 대비: {total_net_usd/total_buy_amt*100 if total_buy_amt else 0:+.2f}%")

    # 미청산 로트
    print(f"\n📦 미청산 매수 로트 (로그 기준):")
    any_open = False
    for sym, q in open_lots.items():
        for qty, px, date in q:
            print(f"   {sym}: {qty}주 @ ${px:,.2f} (매수일 {date})")
            any_open = True
    if not any_open:
        print("   없음")

    # 현재 추적 포지션 파일
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
            pos = json.load(f)
        if pos:
            pos_str = ", ".join(f"{s} {p.get('qty', '?')}주" for s, p in pos.items())
            print(f"\n🤖 봇 추적 포지션: {pos_str}")
    print("=" * 78)


if __name__ == "__main__":
    main()
