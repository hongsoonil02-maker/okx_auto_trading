# -*- coding: utf-8 -*-
"""
position_sizer.py — 실거래 이력 기반 동적 포지션 사이징 (Kelly + 슬리피지/수수료 반영)

P2 목적:
1. 전략의 고정 POSITION_PCT를 실제 성과에 연동된 Kelly 분수로 대체
2. 슬리피지/수수료를 손익비(b)에서 차감하여 "실효 Kelly" 계산
3. 거래 표본이 부족하거나 신뢰도가 낮으면 안전한 기본값으로 폴백 (소표본 과적합 방지)
4. 청산 시 거래를 누적 기록(record_trade)하여 통계 자기학습

설계 안전장치:
- 최소 표본(MIN_SAMPLES) 미만 → 기본 POSITION_PCT 사용
- Quarter Kelly(0.25) 기본 + 최대 상한(MAX_PCT)
- 음수 기대값(불리한 전략) → 0 또는 기본값 하향
- pnl_pct(수익률 %)로 통계 산출 (절대금액 아님)
"""
import os
import csv
import json
import logging
from collections import deque
from datetime import datetime
from typing import Optional

logger = logging.getLogger("position_sizer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIZING_LOG = os.path.join(BASE_DIR, "sizing_trade_log.csv")  # 신전략 거래 누적 (pnl_pct 포함)

# ── 사이징 파라미터 (환경변수로 조정 가능) ──────────────────────────────────
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))   # Quarter Kelly (더 보수적)
MIN_SAMPLES    = int(os.getenv("KELLY_MIN_SAMPLES", "30"))    # 이만큼 쌓이기 전엔 기본값
LOOKBACK_N     = int(os.getenv("KELLY_LOOKBACK_N", "100"))    # 최근 N건으로 통계
MAX_PCT        = float(os.getenv("KELLY_MAX_PCT", "0.25"))    # 종목당 최대 비중 상한 25%
MIN_PCT        = float(os.getenv("KELLY_MIN_PCT", "0.03"))    # 진입 시 최소 비중 3%
MIN_EXPECTANCY_PCT = float(os.getenv("KELLY_MIN_EXPECTANCY_PCT", "0.00"))  # 기대값 하한
MIN_EXPECTANCY_SAMPLES = int(os.getenv("KELLY_MIN_EXPECTANCY_SAMPLES", "20"))

# ── 거래소별 왕복 비용(수수료+슬리피지) %, slippage_fee_engine과 일치 ──────────
ROUNDTRIP_COST_PCT = {
    "OKX":   (0.00 + 0.04) * 2 / 100.0,   # Maker Post-Only 0% + 슬리피지 0.04%, 왕복
    "UPBIT": (0.10 + 0.15) * 2 / 100.0,   # 업비트 taker 0.10% + 슬리피지 0.15%, 왕복
}


def record_trade(market: str, symbol: str, pnl_pct: float, side: str = "EXIT_LONG"):
    """
    청산 거래 1건을 누적 기록. pnl_pct는 수수료/슬리피지 차감 전 가격 기준 % 손익.
    (실효 손익은 통계 계산 시 비용을 차감하여 반영)
    """
    try:
        is_new = not os.path.exists(SIZING_LOG)
        with open(SIZING_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["timestamp", "market", "symbol", "side", "pnl_pct"])
            w.writerow([datetime.now().isoformat(), market.upper(), symbol, side, f"{pnl_pct:.4f}"])
    except Exception as e:
        logger.warning(f"거래 기록 실패: {e}")


def _load_returns(market: str, symbol: Optional[str] = None) -> list:
    """해당 시장(및 선택적 심볼)의 최근 LOOKBACK_N건 pnl_pct(소수, 0.01=1%) 리스트 반환."""
    rets = deque(maxlen=LOOKBACK_N)
    if not os.path.exists(SIZING_LOG):
        return []
    try:
        with open(SIZING_LOG, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("market", "").upper() != market.upper():
                    continue
                if symbol and row.get("symbol", "").upper() != symbol.upper():
                    continue
                try:
                    rets.append(float(row["pnl_pct"]) / 100.0)
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        logger.warning(f"거래 로그 로드 실패: {e}")
        return []
    return list(rets)


def compute_stats(market: str, symbol: Optional[str] = None) -> dict:
    """시장별 또는 시장+심볼별 승률/평균승/평균패(실효, 비용 차감) 계산."""
    rets = _load_returns(market, symbol=symbol)
    n = len(rets)
    cost = ROUNDTRIP_COST_PCT.get(market.upper(), 0.002)

    # 비용 차감한 실효 수익률
    eff = [r - cost for r in rets]
    wins   = [r for r in eff if r > 0]
    losses = [r for r in eff if r < 0]
    nw, nl = len(wins), len(losses)
    win_rate = nw / n if n else 0.0
    avg_win  = (sum(wins) / nw) if nw else 0.0
    avg_loss = (sum(losses) / nl) if nl else 0.0
    return {
        "n": n, "win_rate": win_rate,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "cost_pct": cost,
    }


def expectancy_from_stats(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """기대값 E = p*avg_win + (1-p)*avg_loss. avg_loss는 음수값이어야 함."""
    return (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)


def get_market_expectancy(market: str) -> dict:
    """시장별 최근 기대값과 샘플 수를 반환한다."""
    st = compute_stats(market)
    exp = expectancy_from_stats(st["win_rate"], st["avg_win"], st["avg_loss"])
    return {
        **st,
        "expectancy": exp,
        "expectancy_pct": exp * 100.0,
    }


def get_symbol_expectancy(market: str, symbol: str) -> dict:
    """시장+심볼별 최근 기대값과 샘플 수를 반환한다."""
    st = compute_stats(market, symbol=symbol)
    exp = expectancy_from_stats(st["win_rate"], st["avg_win"], st["avg_loss"])
    return {
        **st,
        "expectancy": exp,
        "expectancy_pct": exp * 100.0,
    }


def is_market_favorable(market: str,
                        min_samples: int = MIN_EXPECTANCY_SAMPLES,
                        min_expectancy_pct: float = MIN_EXPECTANCY_PCT) -> tuple[bool, dict]:
    """
    최근 실현손익 기준으로 신규 진입 허용 여부를 판단한다.

    Returns:
        (allowed, detail)
    """
    st = get_market_expectancy(market)
    allowed = True
    reason = "ok"

    if st["n"] < min_samples:
        reason = "sample_shortage"
    elif st["expectancy_pct"] < min_expectancy_pct:
        # [ORCA-Sizer 가동] 기대값이 일시 음수(-0.1% 등)이더라도 진입을 원천 차단(전체 봇 0건 매매)하지 않고,
        # compute_position_pct에서 최소 탐색 비중(MIN_PCT=3%)으로 통계 회복 매매를 수행하도록 소프트 허용
        allowed = True
        reason = "negative_expectancy_soft_exploration"

    detail = {
        "market": market.upper(),
        "n": st["n"],
        "win_rate": round(st["win_rate"] * 100, 1),
        "avg_win_pct": round(st["avg_win"] * 100, 3),
        "avg_loss_pct": round(st["avg_loss"] * 100, 3),
        "expectancy_pct": round(st["expectancy_pct"], 3),
        "reason": reason,
        "allowed": allowed,
    }
    return allowed, detail


def is_symbol_favorable(market: str,
                        symbol: str,
                        min_samples: int = MIN_EXPECTANCY_SAMPLES,
                        min_expectancy_pct: float = MIN_EXPECTANCY_PCT) -> tuple[bool, dict]:
    """시장+심볼 기준으로 신규 진입 허용 여부를 판단한다."""
    st = get_symbol_expectancy(market, symbol)
    allowed = True
    reason = "ok"

    if st["n"] < min_samples:
        reason = "sample_shortage"
    elif st["expectancy_pct"] < min_expectancy_pct:
        # [ORCA-Sizer 가동] 개별 심볼 기대값 하락 시에도 완전 차단 대신 최소 탐색 비중으로 회복 매매
        allowed = True
        reason = "negative_expectancy_soft_exploration"

    detail = {
        "market": market.upper(),
        "symbol": symbol.upper(),
        "n": st["n"],
        "win_rate": round(st["win_rate"] * 100, 1),
        "avg_win_pct": round(st["avg_win"] * 100, 3),
        "avg_loss_pct": round(st["avg_loss"] * 100, 3),
        "expectancy_pct": round(st["expectancy_pct"], 3),
        "reason": reason,
        "allowed": allowed,
    }
    return allowed, detail


def score_trade_opportunity(
    market: str,
    symbol: str,
    expectancy_pct: float,
    trend_signal: str,
    trend_rsi: float,
    market_regime: str = "NORMAL",
    allowed_side: Optional[str] = None,
    signal: str = "WAIT",
    signal_strength: str = "WEAK",
    momentum_pct: float = 0.0,
    vol_surge: float = 1.0,
    orb_range_pct: float = 0.0,
) -> tuple[float, dict]:
    """
    거래 우선순위를 위한 휴리스틱 점수.

    점수는 절대적인 진입 허용/차단이 아니라, 여러 후보 중 무엇을 먼저 담을지 정하는 데 사용한다.
    기대값, 추세 일치, 모멘텀, 거래량, 신호 강도를 함께 반영한다.
    """
    score = 0.0
    reasons = []

    # 기대값: 1% 기대값 ≈ 10점
    score += expectancy_pct * 10.0
    reasons.append(f"exp={expectancy_pct:+.3f}")

    # 상위 추세 정합성
    trend_signal = (trend_signal or "WAIT").upper()
    side = (allowed_side or "").upper()
    if side == "LONG":
        if trend_signal == "BUY":
            score += 3.0 + max(0.0, trend_rsi - 55.0) * 0.12
        else:
            score -= 5.0
    elif side == "SHORT":
        if trend_signal == "SELL":
            score += 3.0 + max(0.0, 45.0 - trend_rsi) * 0.12
        else:
            score -= 5.0
    reasons.append(f"trend={trend_signal}")

    # 현재 진입 신호와 bias가 맞으면 보너스
    signal = (signal or "WAIT").upper()
    if side and ((side == "LONG" and signal == "LONG") or (side == "SHORT" and signal == "SHORT")):
        score += 1.5
    elif side and signal in ("BUY", "SELL"):
        score += 0.75

    # 모멘텀/거래량/레인지
    score += min(max(momentum_pct * 100.0, 0.0), 5.0) * 0.6
    score += min(max(vol_surge - 1.0, 0.0), 2.5) * 1.5
    score += min(max(orb_range_pct, 0.0), 5.0) * 0.25

    regime = (market_regime or "NORMAL").upper()
    if regime == "EXPANSION":
        score += 2.0
    elif regime == "CONTRACTION":
        score -= 2.5

    if signal_strength.upper() == "STRONG":
        score += 1.0

    detail = {
        "market": market.upper(),
        "symbol": symbol.upper(),
        "score": round(score, 3),
        "expectancy_pct": round(expectancy_pct, 3),
        "trend_signal": trend_signal,
        "trend_rsi": round(float(trend_rsi), 2) if trend_rsi is not None else None,
        "market_regime": regime,
        "allowed_side": side,
        "signal": signal,
        "signal_strength": signal_strength,
        "momentum_pct": round(momentum_pct * 100.0, 3),
        "vol_surge": round(vol_surge, 3),
        "orb_range_pct": round(orb_range_pct, 3),
        "reasons": reasons,
    }
    return score, detail


def kelly_fraction_from_stats(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly f* = (b*p - q)/b, b=avg_win/|avg_loss|. 음수는 0."""
    if avg_loss == 0 or avg_win <= 0:
        return 0.0
    b = avg_win / abs(avg_loss)
    if b <= 0:
        return 0.0
    p = win_rate
    q = 1.0 - win_rate
    kelly = (b * p - q) / b
    return max(0.0, kelly)


def compute_position_pct(market: str, default_pct: float) -> tuple:
    """
    시장별 동적 포지션 비중(0~1) 계산.
    Returns: (position_pct, detail_dict)

    - 표본 < MIN_SAMPLES: default_pct 사용 (신뢰 부족)
    - Kelly 0 (불리): MIN_PCT로 축소 (또는 default와 작은 값)
    - 그 외: Quarter Kelly, [MIN_PCT, MAX_PCT]로 클램프
    """
    st = compute_stats(market)
    n = st["n"]

    if n < MIN_SAMPLES:
        return default_pct, {
            "mode": "default(표본부족)", "n": n, "need": MIN_SAMPLES,
            "pct": default_pct,
        }

    full_kelly = kelly_fraction_from_stats(st["win_rate"], st["avg_win"], st["avg_loss"])
    frac = full_kelly * KELLY_FRACTION  # Quarter Kelly

    if frac <= 0:
        # 통계적으로 불리 → 최소 비중으로 방어 (또는 진입 자제는 전략 레벨에서)
        pct = min(default_pct, MIN_PCT)
        mode = "kelly<=0(방어)"
    else:
        pct = max(MIN_PCT, min(MAX_PCT, frac))
        mode = "kelly"

    return pct, {
        "mode": mode, "n": n,
        "win_rate": round(st["win_rate"], 3),
        "avg_win": round(st["avg_win"], 4),
        "avg_loss": round(st["avg_loss"], 4),
        "full_kelly": round(full_kelly, 4),
        "quarter": round(frac, 4),
        "cost_pct": round(st["cost_pct"], 4),
        "pct": round(pct, 4),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for mk, dft in [("OKX", 0.20), ("UPBIT", 0.25)]:
        pct, detail = compute_position_pct(mk, dft)
        print(f"[{mk}] 사이징 = {pct*100:.1f}%  ({detail})")
